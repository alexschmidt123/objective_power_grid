"""One bounded, persistent, sequential development pilot; never promotes results."""
import time
ENTRY=time.monotonic()
import argparse,copy,fcntl,hashlib,json,math,os,signal,subprocess,traceback,tarfile
from pathlib import Path
import numpy as np
import torch,yaml
from src.research.moe_sboed_v3 import ProcessMoE,dense_match,process_views,ConditionalMixture
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.journal_validation import CommonValidation

ROOT=Path(__file__).resolve().parents[2]
ARMS=['v3','v1','dense','shared','noaux']
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def summary(x):
    x=np.asarray(x);mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)))
    return dict(mean=mean,MC_SE=se,CI95=[mean-1.96*se,mean+1.96*se],N=len(x))
def sources():
    paths=[ROOT/'run.sh',ROOT/'sweep_run.sh']
    for folder in ['src','tools','configs','scripts']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.sh'] and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
def check_sources(manifest):
    for rel,digest in manifest.items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen source changed: '+rel)

class BudgetEnd(Exception):pass
def alarm(*args):raise BudgetEnd('7200-second training budget boundary')

class Intervention(torch.nn.Module):
    """Single-decision substitution after a common unmodified observed prefix."""
    kind='moe_sboed_v3';outputs_design=False
    def __init__(self,policy,expert,stage=15):super().__init__();self.policy=policy;self.expert=expert;self.stage=stage
    def forward(self,f,stage):
        self.policy.substitution=self.expert if stage==self.stage else None
        return self.policy(f,stage)

@torch.no_grad()
def calibration_and_context(model,validator,C,tag):
    rows=[];contexts=[];util=torch.zeros(4,dtype=torch.float64);lat=[];p=model.eval()
    for start in range(0,validator.systems,16):
        n=min(16,validator.systems-start)
        with torch.device('cuda:0'):
            b=make_benchmark(validator.raw);r=rollout(b,p,30,n,validator.seed+start*97451,0,truth=validator.truth[start:start+n].cuda(),noise=validator.noise[start:start+n].cuda())
            target=b.transform(r['truth'])
            for stage in [0,5,15,29]:
                f=r['features'][stage];dist=p.uncertainty(p.adapter(f,stage));w,mu,L=dist;views=process_views(dist)
                marginal_sd=L.square().sum(-1).sqrt();cdf=.5*(1+torch.erf((target[:,None]-mu)/(marginal_sd*math.sqrt(2))));pit=(w[:,:,None]*cdf).sum(1)
                lp=ConditionalMixture.log_prob(dist,target)
                # Two-source permutation symmetry is a domain-specific audit,
                # not part of the general architecture or a training target.
                swapped=target[:,[2,3,0,1]];gap=(lp-ConditionalMixture.log_prob(dist,swapped)).abs()
                rows.append(dict(stage=stage,N=n,NLL=float(-lp.mean()),PIT=pit.cpu().tolist(),swap_log_density_gap=float(gap.mean())))
                if stage==15:
                    Ck=L@L.transpose(-1,-2);mean=(w[:,:,None]*mu).sum(1);delta=mu-mean[:,None]
                    within=(w*Ck.diagonal(dim1=-2,dim2=-1).sum(-1)).sum(-1)
                    between=(w*delta.square().sum(-1)).sum(-1)
                    contexts.extend(torch.stack((within+between,between/(within+between)),1).cpu().tolist())
                p(f,stage);ids=p.last_dispatch['indices'].cpu();util+=torch.nn.functional.one_hot(ids,4).sum((0,1)).double()
            # Batched latency is a throughput diagnostic, not individual GPU latency.
            torch.cuda.synchronize();t=time.perf_counter()
            for _ in range(10):p(r['features'][15],15)
            torch.cuda.synchronize();lat.append((time.perf_counter()-t)/10/n)
    report={}
    for stage in [0,5,15,29]:
        part=[x for x in rows if x['stage']==stage];pit=np.concatenate([x['PIT'] for x in part]);ks=[]
        for j in range(pit.shape[1]):
            v=np.sort(pit[:,j]);n=len(v);ks.append(float(max(np.max(np.arange(1,n+1)/n-v),np.max(v-np.arange(n)/n))))
        report[str(stage)]=dict(NLL=float(np.mean([x['NLL'] for x in part])),marginal_PIT_KS=ks,central90_coverage=((pit>=.05)&(pit<=.95)).mean(0).tolist(),mean_source_swap_log_density_gap=float(np.mean([x['swap_log_density_gap'] for x in part])))
    write(C/'evaluations'/f'{tag}_density_audit.json',dict(stages=report,expert_slot_utilization=(util/util.sum()).tolist(),batch16_seconds_per_decision=float(np.mean(lat)),limitation='Marginal calibration and source permutation symmetry do not establish full multivariate posterior fidelity. Mixture-mode count is not posterior-mode evidence.'))
    return np.asarray(contexts)

def evaluate(models,updates,C,cfg,seed,N,L,tag,diagnostics=False):
    validator=CommonValidation(cfg['benchmark'],30,seed,N,L,batch=16,chunk=16384,device='cuda:0')
    # Preserve actual independent development cases, not only a seed.
    np.savez(C/'evaluations'/f'{tag}_cases.npz',truth=validator.truth.numpy(),noise=validator.noise.numpy(),contrasts=validator.bank.numpy())
    results={};scores={}
    for arm,model in models.items():
        model.cuda().eval();x,meta=validator(model);scores[arm]=x;results[arm]=dict(**summary(x),update=updates[arm],metadata=meta)
        np.save(C/'evaluations'/f'{tag}_{arm}.npy',x)
    results['paired_v3_minus_v1']=summary(scores['v3']-scores['v1'])
    results['matched_updates']=len(set(updates.values()))==1
    if diagnostics:
        p=models['v3'];contexts=calibration_and_context(p,validator,C,tag)
        interventions={}
        for mode in ['uniform','random','mask']:
            q=copy.deepcopy(p);q.routing_mode=mode if mode!='mask' else 'learned';q.mask_routed=mode=='mask'
            x,meta=validator(q);np.save(C/'evaluations'/f'{tag}_{mode}.npy',x)
            interventions[mode]=dict(full_minus_intervention=summary(scores['v3']-x),metadata=meta)
        # Frozen, single-stage branches share the complete prefix through t=15.
        branch=[]
        for e in range(4):
            x,meta=validator(Intervention(copy.deepcopy(p),e));np.save(C/'evaluations'/f'{tag}_substitute{e}.npy',x);branch.append(x)
        tests=[]
        for feature in range(2):
            cut=float(np.median(contexts[:,feature]));low=contexts[:,feature]<=cut
            for a in range(4):
                for b in range(a):
                    delta=branch[a]-branch[b];lo=summary(delta[low]);hi=summary(delta[~low]);effect=hi['mean']-lo['mean'];se=math.hypot(hi['MC_SE'],lo['MC_SE'])
                    z=effect/se if se else 0.;pvalue=math.erfc(abs(z)/math.sqrt(2))
                    tests.append(dict(feature=['total_variance','between_fraction'][feature],threshold=cut,a=a,b=b,low=lo,high=hi,interaction=effect,SE=se,pvalue=pvalue))
        order=sorted(range(len(tests)),key=lambda i:tests[i]['pvalue']);running=0.
        for rank,i in enumerate(order):running=max(running,min(1.,(len(tests)-rank)*tests[i]['pvalue']));tests[i]['Holm_pvalue']=running
        results['interventions']=interventions;results['conditional_competence']=tests
    write(C/'evaluations'/f'{tag}_summary.json',results)
    text='DEVELOPMENT_UNVALIDATED. Finite-contrast paired development estimates; one training seed.\n\n'
    text+='| Arm | Updates | EIG (nats) | MC SE |\n|---|---:|---:|---:|\n'
    for arm in models:text+=f"| {arm} | {updates[arm]} | {results[arm]['mean']:.6f} | {results[arm]['MC_SE']:.6f} |\n"
    d=results['paired_v3_minus_v1'];text+=f"\nPaired v3−v1: {d['mean']:.6f} ± {d['MC_SE']:.6f}; 95% CI {d['CI95']}. Matched updates: {results['matched_updates']}.\n"
    text+='No official seed1001 draws used. MC SE conditions on the common finite contrast bank and does not cover training-seed variability or contrast bias.\n'
    (ROOT/'reports/MOE_SBOED_V3_EIG.md').write_text(text)
    write(C/'latest_evaluation.json',dict(tag=tag,results=results))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--moe-sboed-v3-development',action='store_true');p.add_argument('--config',default='configs/moe_sboed_v3_development.yaml');p.add_argument('--output',required=True);args=p.parse_args()
    if ROOT.name!='objective_power_grid_temp':raise RuntimeError('V3 development may execute only in TEMP')
    from tools.storage_policy import enforce_storage_policy
    enforce_storage_policy(ROOT,['--output',args.output]);C=Path(args.output).resolve();C.mkdir(parents=True,exist_ok=False)
    for d in ['models','logs','evaluations','provenance']:(C/d).mkdir()
    cfg=yaml.safe_load((ROOT/args.config).read_text());cfg['benchmark']=yaml.safe_load((ROOT/'configs/deepseek_sboed_location.yaml').read_text())['benchmark'];write(C/'effective_config.json',cfg)
    locks=[]
    for path in [ROOT.parent/'objective_power_grid/.gpu0.lock',Path.home()/'.cache/objective_power_grid/device_0.lock']:
        path.parent.mkdir(parents=True,exist_ok=True);f=open(path,'a+')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:write(C/'status.json',dict(state='GPU_BUSY_NOT_LAUNCHED',lock=str(path)));return
        locks.append(f)
    existing=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if existing:write(C/'status.json',dict(state='GPU_BUSY_NOT_LAUNCHED',pids=existing));return
    manifest=sources();write(C/'provenance/source_sha256.json',manifest)
    with tarfile.open(C/'provenance/frozen_source.tar.gz','w:gz') as archive:
        for rel in manifest:archive.add(ROOT/rel,arcname=rel,recursive=False)
    write(C/'provenance/git.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),working_tree=subprocess.check_output(['git','status','--short'],text=True),archive_sha256=sha(C/'provenance/frozen_source.tar.gz')))
    receipt=Path('/tmp/moe_v3_correctness.xml')
    if not receipt.is_file():raise RuntimeError('Missing CPU readiness receipt')
    (C/'provenance/correctness.xml').write_bytes(receipt.read_bytes())
    torch.set_num_threads(4);torch.cuda.init();torch.cuda.synchronize();setup=time.monotonic()-ENTRY
    state={};models={};opts={};schedulers={};history={arm:[] for arm in ARMS}
    signal.signal(signal.SIGALRM,alarm)
    for arm in ARMS:
        t=time.monotonic();torch.manual_seed(cfg['training_seed'])
        if arm=='v1':m=DeepSeekPolicy()
        elif arm=='dense':m=dense_match(models['v3'].counts()['total_parameters'])
        else:m=ProcessMoE(variant={'v3':'full','shared':'shared','noaux':'noaux'}[arm])
        models[arm]=m.cuda();opts[arm]=torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998));schedulers[arm]=torch.optim.lr_scheduler.ExponentialLR(opts[arm],.98)
        torch.cuda.synchronize();state[arm]=dict(updates=0,charged_seconds=setup+time.monotonic()-t,max_cycle_seconds=1.,status='READY',fresh_episodes=0,scalar_likelihood_evaluations=0,auxiliary_pairs=0,counts=m.counts())
    def status(active,phase):
        record=dict(state=phase,PID=os.getpid(),active_method=active,updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),arms=state,training_cap_per_arm_seconds=cfg['training_seconds_per_arm'],source_manifest_sha256=sha(C/'provenance/source_sha256.json'))
        write(C/'status.json',record);write(ROOT/'reports/MOE_SBOED_V3_CURRENT.json',dict(output=str(C),**record))
    def checkpoint(arm):
        m=models[arm];t=time.monotonic();path=C/'models'/f'{arm}_update{state[arm]["updates"]}.pt';tmp=path.with_suffix('.pt.tmp')
        torch.save(dict(method='moe_sboed_v3' if arm=='v3' else arm,status='DEVELOPMENT_UNVALIDATED',policy=copy.deepcopy(m).cpu(),optimizer=opts[arm].state_dict(),scheduler=schedulers[arm].state_dict(),state=state[arm],effective_configuration=cfg,source_manifest_sha256=sha(C/'provenance/source_sha256.json')),tmp);tmp.replace(path)
        state[arm]['charged_seconds']+=time.monotonic()-t
        if state[arm]['charged_seconds']>cfg['training_seconds_per_arm']:raise BudgetEnd('Checkpoint serialization exceeded budget; previous checkpoint remains eligible')
        write(C/'models'/f'{arm}_latest.json',dict(path=str(path),sha256=sha(path),**state[arm]))
    try:
        for milestone in cfg['milestones']:
            check_sources(manifest)
            for arm in ARMS:
                s=state[arm];m=models[arm];opt=opts[arm]
                if s['status'] in ['BUDGET_COMPLETE','FAILED']:continue
                m.train();b=make_benchmark(cfg['benchmark']);s['status']='TRAINING';status(arm,'RUNNING')
                while s['updates']<milestone:
                    remaining=cfg['training_seconds_per_arm']-s['charged_seconds']
                    if remaining<max(30.,2*s['max_cycle_seconds']+15):s['status']='BUDGET_COMPLETE';break
                    signal.setitimer(signal.ITIMER_REAL,remaining-15);t=time.monotonic();update=s['updates']+1
                    try:
                        opt.zero_grad(set_to_none=True)
                        with torch.device('cuda:0'):
                            r=rollout(b,m,30,cfg['batch_size'],cfg['training_seed']+update*104729,cfg['contrasts']);info=r['info'][:,-1].mean()
                            aux=m.auxiliary_on(r['features'],b.transform(r['truth'])) if arm in ['v3','dense','shared'] else info*0
                            loss=-info+(m.balance_loss() if arm=='v1' else cfg['auxiliary_weight']*aux)
                            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                            loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
                            if (update-1)%1000==0:schedulers[arm].step()
                            torch.cuda.synchronize()
                        dt=time.monotonic()-t;s['charged_seconds']+=dt;s['max_cycle_seconds']=max(s['max_cycle_seconds'],dt);s['updates']=update;s['fresh_episodes']+=cfg['batch_size'];s['scalar_likelihood_evaluations']+=cfg['batch_size']*(cfg['contrasts']+1)*30
                        if arm in ['v3','dense','shared']:s['auxiliary_pairs']+=cfg['batch_size']*30
                        if update==1:
                            safe=min(cfg['milestones'][-1],int((cfg['training_seconds_per_arm']-s['charged_seconds']-30)/(1.5*dt))+1)
                            write(C/'logs'/f'{arm}_measured_schedule.json',dict(first_native_cycle_seconds=dt,conservative_max_updates=safe,native_optimizer='Adam',B=2000,L=2000,horizon=30,lr=5e-5,betas=[.8,.998],schedule='gamma .98 after updates 1,1001,...',selection='final eligible checkpoint; no test EIG selection'))
                        if update==1 or update%25==0:
                            row=dict(update=update,charged_seconds=s['charged_seconds'],train_sPCE=float(info.detach()),auxiliary_NLL_per_dimension=float(aux.detach()),gradient_norm=float(gn),cycle_seconds=dt,lr=opt.param_groups[0]['lr']);history[arm].append(row)
                            with open(C/'logs'/f'{arm}.jsonl','a') as f:f.write(json.dumps(row)+'\n')
                            print(arm,row,flush=True);status(arm,'RUNNING')
                        if update==1 or update%256==0:signal.setitimer(signal.ITIMER_REAL,0);checkpoint(arm)
                    except BudgetEnd:
                        s['charged_seconds']+=time.monotonic()-t;s['status']='BUDGET_COMPLETE';break
                    finally:signal.setitimer(signal.ITIMER_REAL,0)
                if s['status']!='BUDGET_COMPLETE':s['status']='MILESTONE_COMPLETE';checkpoint(arm)
                elif not (C/'models'/f'{arm}_latest.json').exists():raise RuntimeError('Budget exhausted before any eligible checkpoint')
            common=min(x['updates'] for x in state.values())
            # Only score checkpoints at the identical declared milestone.
            if all(s['updates']>=milestone for s in state.values()):
                status(None,'EVALUATING');updates={a:milestone for a in ARMS}
                evaluate(models,updates,C,cfg,cfg['development_seed'],256,65536,f'update{milestone}')
            if any(s['status']=='BUDGET_COMPLETE' for s in state.values()):break
        # Highest milestone completed by ALL arms, never selected by EIG.
        eligible=[n for n in cfg['milestones'] if all((C/'models'/f'{a}_update{n}.pt').is_file() for a in ARMS)]
        if not eligible:status(None,'INCONCLUSIVE_NO_COMMON_MILESTONE');return
        chosen=max(eligible);final={}
        for a in ARMS:final[a]=torch.load(C/'models'/f'{a}_update{chosen}.pt',map_location='cuda:0',weights_only=False)['policy'].eval()
        status(None,'FINAL_DEVELOPMENT_EVALUATION');updates={a:chosen for a in ARMS}
        result=evaluate(final,updates,C,cfg,cfg['development_seed']+1,512,262144,f'final_update{chosen}',diagnostics=True)
        d=result['paired_v3_minus_v1'];decision='NEGATIVE' if d['CI95'][1]<0 else 'INCONCLUSIVE'
        if d['CI95'][0]>0:
            confirm=evaluate(final,updates,C,cfg,cfg['development_seed']+2,512,262144,f'confirmation_update{chosen}',diagnostics=True)
            decision='DEVELOPMENT_GAIN_REPLICATED' if confirm['paired_v3_minus_v1']['CI95'][0]>0 else 'INCONCLUSIVE_CONFIRMATION'
        write(C/'decision.json',dict(decision=decision,matched_update=chosen,permanent_promotion=False,limitation='One training seed; posterior surrogate and expert-role attribution require inspection. No cross-domain EIG claim.'))
        status(None,'DEVELOPMENT_COMPLETE')
    except Exception as e:
        write(C/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),arms=state));status(None,'FAILED_NO_AUTOMATIC_RESTART');raise

if __name__=='__main__':main()
