"""One teacher-free SA-MoE/control experiment through the canonical launcher."""
import argparse,copy,csv,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch,yaml
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.policies import history_features
from src.research.sa_moe import SAMoE,load_sa_checkpoint
from src.research.sa_moe_offline import posterior_from_history,measure_competence,conditional_rollout
from src.research.pimoe_experiment import write,sha


def record(C,name,obj):write(C/name,obj)

def save_model(model,C,name,update,extra):
    state=dict(method='SA-MoE-sBOED',single=model.single,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},training_seed=101,update=update,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),**extra)
    path=C/'models'/f'{name}.pt';torch.save(state,path);return path


def bootstrap_contexts(model,result,opt,C):
    features=torch.cat([result['features'][t].detach() for t in [4,10,18,26]])
    with torch.no_grad():
        z=model.context(features);x=(z-z.mean(0))/z.std(0,unbiased=False).clamp_min(.1);centers=[x[0]]
        for _ in range(3):centers.append(x[torch.cdist(x,torch.stack(centers)).min(1).values.argmax()])
        centers=torch.stack(centers)
        for _ in range(10):
            assignment=torch.cdist(x,centers).argmin(1)
            for e in range(4):
                if (assignment==e).any():centers[e]=x[assignment==e].mean(0)
    for _ in range(100):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(model.router(model.context(features)),assignment);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    record(C,'reports/bootstrap_contexts.json',dict(assignments=assignment.cpu().tolist(),stage_order=[4,10,18,26],note='Initial learned-context diversification only; no permanent role labels or equal-use penalty'))


def build_contexts(cfg,result,seed):
    cpu_b=make_benchmark(cfg['benchmark']);cpu_b.pimoe_work={};contexts=[]
    for k,t in enumerate([4,10,18,26]):
        actions=[a[0].detach().cpu() for a in result['actions'][:t]];ys=[y[0].detach().cpu() for y in result['observations'][:t]]
        belief=posterior_from_history(cpu_b,actions,ys,seed+k*101,particles=2048,refined=8192)
        f=result['features'][t][0:1].detach().cpu()
        contexts.append(dict(actions=actions,observations=ys,posterior=belief.theta.detach().cpu(),features=f,stage=t,posterior_diagnostics=belief.records))
    return contexts,dict(forward_parameter_stages=cpu_b.calls,likelihood_terms=cpu_b.pimoe_work.get('scalar_conditional_likelihood_evaluations'),fresh_observed_history_source='prefix of recorded full simulation; not counted twice',conditional_SMC_particles=8192,SMC_not_exact=True)


def specialize(b,model,opt,contexts,competence,C,seed,single=False):
    if not single:
        log_r=torch.tensor([r['log_responsibilities'] for r in competence],dtype=torch.float64,device='cuda')
        training_weight=log_r.log_softmax(0).exp()/4
    opt.zero_grad();total=0.
    with torch.device('cuda'):
        for i,ctx in enumerate(contexts):
            for e in range(4):
                value=conditional_rollout(b,model,ctx['actions'],ctx['observations'],ctx['posterior'],0 if single else e,seed+i*101,batch=8,contrasts=8192)
                weight=1/(len(contexts)*4) if single else training_weight[i,e]
                term=-weight*value.mean();term.backward();total+=float(term.detach())
            if not single:
                target=torch.tensor(competence[i]['responsibilities'],dtype=torch.float64,device='cuda')
                f=ctx['features'].to('cuda');ce=-(target*model.router(model.context(f)).log_softmax(-1)[0]).sum()/len(contexts)
                ce.backward();total+=float(ce.detach())
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();torch.cuda.synchronize()
    return dict(loss=total,gradient_norm=norm,conditional_expert_pathwise_updates=4*len(contexts),conditional_trajectories=8*4*len(contexts),contrasts=8192,shared_training_contexts=True)


def train_one(C,cfg,shared,deadline,single=False):
    name='single' if single else 'sa_moe';torch.manual_seed(101);model=SAMoE(single).cuda();model.train();opt=torch.optim.Adam(model.parameters(),lr=.0004)
    b=make_benchmark(cfg['benchmark']);start=time.perf_counter();records=[];last=None;specializations=[];bootstrap_done=False;period=cfg['competence_updates']
    for update in range(1,cfg['maximum_updates']+1):
        if time.perf_counter()-start>cfg['per_student_seconds']-5 or time.perf_counter()>deadline-60:break
        unit=time.perf_counter();opt.zero_grad()
        with torch.device('cuda'):
            result=rollout(b,model,30,cfg['batch_size'],10100000+update,cfg['training_contrasts'])
            info=result['info'][:,-1];loss=-info.mean();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();torch.cuda.synchronize()
        last=update
        if update==100 and not single:
            bootstrap_contexts(model,result,opt,C);bootstrap_done=True
        if update in period and time.perf_counter()<deadline-120:
            begin=time.perf_counter()
            if single:
                if str(update) in shared:
                    contexts,comp=shared[str(update)];aux=specialize(b,model,opt,contexts,comp,C,10200000+update,single=True)
                else:aux=dict(status='Main run did not produce this context batch')
            else:
                with torch.device('cpu'):contexts,posterior_work=build_contexts(cfg,result,10300000+update)
                comp=measure_competence(b,model,contexts,10400000+update,batch=32,contrasts=8192)
                shared[str(update)]=(contexts,comp);torch.save(dict(contexts=contexts,competence=comp,posterior_work=posterior_work),C/'evaluations'/f'competence_{update}.pt')
                aux=specialize(b,model,opt,contexts,comp,C,10200000+update)
                record(C,f'reports/competence_{update}.json',dict(contexts=comp,posterior_work=posterior_work,training_only=True))
            specializations.append(dict(update=update,seconds=time.perf_counter()-begin,**aux))
        if update==1 or update%100==0:
            with torch.no_grad():
                z=model.context(result['features'][15].detach());p=model.probabilities(z);mass=p.mean(0);effective=(-(p*p.clamp_min(1e-300).log()).sum(-1)).exp().mean()
            row=dict(update=update,terminal_training_sPCE=float(info.mean()),training_bound_ceiling=math.log(cfg['training_contrasts']+1),gradient_norm=norm,elapsed_seconds=time.perf_counter()-start,update_seconds=time.perf_counter()-unit,soft_expert_mass=mass.cpu().tolist(),mean_effective_experts=float(effective),fresh_complete_primary_episodes=update*cfg['batch_size'])
            records.append(row);print(name,row,flush=True);record(C,f'costs/{name}_progress.json',dict(records=records,specializations=specializations,forward_parameter_stages=b.calls,CPU_process_seconds=time.process_time()))
            if time.perf_counter()-start<cfg['per_student_seconds']:save_model(model,C,name+'_progress',update,dict(checkpoint_selection='latest within declared student phase allowance'))
    if last is None:raise RuntimeError('No eligible native update')
    elapsed=time.perf_counter()-start
    if elapsed>cfg['per_student_seconds']:
        path=C/'models'/f'{name}_progress.pt'
        if not path.exists():raise RuntimeError('Overshot phase allowance with no eligible progress checkpoint')
        model=load_sa_checkpoint(path,'cuda');selected=torch.load(path,map_location='cpu',weights_only=False)['update'];stop='phase_cap_overshoot_selected_last_eligible'
    else:selected=last;stop='reference_update_schedule_completed' if last==cfg['maximum_updates'] else 'phase_compute_allowance'
    path=save_model(model,C,name,selected,dict(stop_reason=stop,charged_phase_seconds=elapsed))
    record(C,f'costs/{name}_final.json',dict(elapsed_seconds=elapsed,selected_update=selected,executed_update=last,primary_episodes=last*cfg['batch_size'],main_true_parameter_stages=last*cfg['batch_size']*30,main_contrast_parameter_stages=last*cfg['batch_size']*cfg['training_contrasts']*30,main_scalar_likelihood_terms=last*cfg['batch_size']*(cfg['training_contrasts']+1)*30,all_phase_forward_parameter_stages=b.calls,additional_conditional_work=specializations,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(path),stop_reason=stop,GPU_active_seconds=None,wall_time_unit='exclusive-device phase elapsed time, not GPU-active time',inherited_pretraining_seconds=0))
    return model.eval()


def development(C,cfg,moe,single,deadline):
    b=make_benchmark(cfg['benchmark']);rows=[];state_rows=[];dev_start=time.perf_counter()
    for mode,model in [('learned',moe),('uniform',moe),('single',single)]:
        model.routing_mode='uniform' if mode=='uniform' else 'learned';model.eval();model.inference_times=[];torch.manual_seed(1013)
        for rep in range(4):
            if time.perf_counter()>deadline-5:raise TimeoutError('Development incomplete under declared additional-compute ceiling')
            with torch.no_grad(),torch.device('cuda'):
                r=rollout(b,model,30,8,10130000+rep,16384)
            rows.append(dict(mode=mode,replicate_batch=rep,mean_sPCE=float(r['info'][:,-1].mean()),trajectory_scores=r['info'][:,-1].cpu().tolist(),contrasts=16384,latencies=model.inference_times[-30:]))
            if mode=='learned' and rep==0:
                with torch.no_grad():
                    for t in range(30):
                        z=model.context(r['features'][t]);proposals=model.all_proposals(z);p=model.probabilities(z);state_rows.append(dict(stage=t,soft_mass=p.mean(0).cpu().tolist(),hard_frequency=torch.bincount(p.argmax(-1),minlength=4).double().div(len(p)).cpu().tolist(),mean_pairwise_action_distance=float(torch.cdist(proposals,proposals).mean()),router_entropy=float(-(p*p.clamp_min(1e-300).log()).sum(-1).mean())))
    moe.routing_mode='learned';record(C,'reports/development_comparison.json',dict(rows=rows,expert_by_stage=state_rows,seed=1013,official_seed1001_used=False,finite_contrast_limit=math.log(16385),warning='Development sPCE is contrast-limited; no asymptotic EIG or statistical training-seed claims'))
    numeric=all(np.isfinite(row['trajectory_scores']).all() for row in rows)
    if not numeric:raise FloatingPointError('Nonfinite development information; official evaluation blocked')
    # Independent held-out contexts, never used for gradients or router updates.
    with torch.no_grad(),torch.device('cuda'):
        heldout=rollout(b,moe,30,1,10139991,0)
    with torch.device('cpu'):contexts,work=build_contexts(cfg,heldout,10500001)
    competence=measure_competence(b,moe,contexts,10510001,batch=32,contrasts=8192)
    comparison=[]
    with torch.no_grad(),torch.device('cuda'):
        for i,ctx in enumerate(contexts):
            values=conditional_rollout(b,single,ctx['actions'],ctx['observations'],ctx['posterior'],0,10510001+i*101,batch=32,contrasts=8192)
            f=ctx['features'].to('cuda');chosen=int(moe.probabilities(moe.context(f)).argmax())
            q=competence[i]['Q'];winner=int(np.argmax(q));second=int(np.argsort(q)[-2]);delta=q[winner]-q[second];se=competence[i]['paired_difference_SE'][winner][second]
            comparison.append(dict(stage=ctx['stage'],learned_expert=chosen,highest_estimated_expert=winner,resolved_winner=delta>2*se and not competence[i]['saturated'],routing_regret_lower_bound=q[winner]-q[chosen],uniform_first_action_value=float(np.mean(q)),single_remaining_information=float(values.mean()),single_MC_SE=float(values.std(unbiased=True)/math.sqrt(len(values))),competence=competence[i]))
    record(C,'reports/heldout_expert_competence.json',dict(contexts=comparison,posterior_work=work,training_updates_on_these_states=0,MC_error_excludes_SMC_bias=True))
    record(C,'costs/development.json',dict(seconds=time.perf_counter()-dev_start,forward_parameter_stages=b.calls,posterior_work=work,full_validation_trajectories=97,conditional_rollout_work='Included in forward_parameter_stages; not primary episodes',GPU_active_seconds=None))
    return dict(PASS=numeric,numerical=numeric,offline_only=True,selection='Latest phase-budget-eligible model, not maximum development EIG',official_seed_used=False)


def run_train(C,cfg):
    from tools.journal_campaign import exclusive_device_lock
    lock=exclusive_device_lock();started=time.perf_counter();deadline=started+cfg['total_new_experimental_seconds'];shared={};status='FAILED';torch.set_num_threads(2)
    try:
        moe=train_one(C,cfg,shared,deadline);single=train_one(C,cfg,shared,deadline,single=True)
        gate=development(C,cfg,moe,single,deadline);record(C,'readiness/development_gate.json',gate)
        record(C,'reports/checkpoint_freeze.json',dict(main_sha256=sha(C/'models/sa_moe.pt'),single_sha256=sha(C/'models/single.pt'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),selection='Latest eligible checkpoints; development never official evaluation'))
        status='READY_FOR_OFFICIAL_EVALUATION'
    except Exception as exc:
        record(C,'reports/failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:
        record(C,'registry.json',dict(status=status,new_experimental_seconds=time.perf_counter()-started,ceiling_seconds=cfg['total_new_experimental_seconds'],hardware=torch.cuda.get_device_name(0),no_inherited_model=True,classification='ADDITIONAL_COMPUTE_NOT_MATCHED_TO_IC',CPU_process_seconds=time.process_time()));print('SA_MOE_STATUS',status,flush=True)


def official(C,cfg,which):
    from tools.journal_campaign import exclusive_device_lock
    from src.research.sir_evaluation import contrast_scores
    lock=exclusive_device_lock();path=C/'models'/('sa_moe.pt' if which=='main' else 'single.pt');freeze=json.loads((C/'reports/checkpoint_freeze.json').read_text());expected=freeze['main_sha256' if which=='main' else 'single_sha256']
    if sha(path)!=expected:raise ValueError('Frozen checkpoint changed')
    canonical=Path(cfg['canonical_evaluation_directory'])
    for filename,key in [('location_cases.npz','cases_sha256'),('location_contrasts.npy','contrast_sha256')]:
        if sha(canonical/filename)!=cfg[key]:raise ValueError('Canonical bank mismatch')
    model=load_sa_checkpoint(path,'cpu');b=make_benchmark(cfg['benchmark']);cases=np.load(canonical/'location_cases.npz');truth=torch.as_tensor(cases['truth'],dtype=torch.float64);noise=torch.as_tensor(cases['noise'],dtype=torch.float64);bank=torch.as_tensor(np.load(canonical/'location_contrasts.npy'),dtype=torch.float64)
    out=C/'evaluations'/which;out.mkdir(exist_ok=True);journal=out/'trajectory_records.jsonl';scores=[];start=time.perf_counter()
    if (out/'result.json').exists():raise RuntimeError('Completed official evaluation preserved')
    if journal.exists():
        for line in journal.open():
            row=json.loads(line);assert row['checkpoint_sha256']==expected and row['index']==len(scores);scores.append(row['spce'])
    for i in range(len(scores),2000):
        with torch.no_grad():r=rollout(b,model,30,1,9000000+i,0,truth=truth[i:i+1],noise=noise[i:i+1],track_likelihood=True)
        info,_=contrast_scores(b,r,bank,4096);score=float(info[0,-1]);scores.append(score)
        row=dict(index=i,spce=score,checkpoint_sha256=expected,decision_seconds=r['decision_seconds'],neural_component_times=model.inference_times[-30:],belief_update_seconds=0.,belief_update_definition='No numerical posterior updater; measured history is encoded in next shared forward',truth=truth[i].tolist(),noise=noise[i].tolist())
        with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        if (i+1)%10==0:record(C,f'evaluations/{which}/progress.json',dict(trajectories=i+1,total=2000,seconds=time.perf_counter()-start));print(which,'evaluation',i+1,flush=True)
    np.save(out/'scores.npy',scores);result=dict(method='SA-MoE-sBOED' if which=='main' else 'parameter_matched_single',EIG=float(np.mean(scores)),MC_SE=float(np.std(scores,ddof=1)/math.sqrt(2000)),training_seed=101,evaluation_seed=1001,trajectories=2000,contrasts=1000000,horizon=30,checkpoint_sha256=expected,evaluation_seconds_this_session=time.perf_counter()-start,cases_sha256=cfg['cases_sha256'],contrast_sha256=cfg['contrast_sha256'])
    record(C,f'evaluations/{which}/result.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sa-moe',action='store_true');p.add_argument('--action',choices=['train','evaluate'],required=True);p.add_argument('--campaign',required=True);p.add_argument('--config',required=True);p.add_argument('--which',choices=['main','single'],default='main');a=p.parse_args();C=Path(a.campaign).resolve();cfg=yaml.safe_load(Path(a.config).read_text());torch.set_num_threads(2)
    if a.action=='train':run_train(C,cfg)
    else:official(C,cfg,a.which)
if __name__=='__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: archived exploratory entry point')
if False:main()
