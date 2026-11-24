"""V5-A empirical-SMC conditional auxiliary; one guarded GPU seed at a time."""
import time
ENTRY=time.monotonic()
from pathlib import Path
import argparse,copy,csv,fcntl,hashlib,importlib.util,json,math,os,signal,subprocess,sys,traceback
import numpy as np
import torch,yaml
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.engine import rollout,generator
from src.research.benchmarks import make_benchmark
from src.research.compute_budget import ComputeClock
from src.research.moe_v5_experiment import verified_reference,load_policy
from src.research.moe_v5a_sampler import conditional_auxiliary

VARIANT='V5-A_empirical_SMC_detached_gradient'
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def acquire():
    handles=[]
    try:
        for path in [ROOT/'.gpu0.lock',ROOT.parent/'objective_power_grid/.gpu0.lock',Path.home()/'.cache/objective_power_grid/device_0.lock']:
            f=path.open('a+');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy:raise BlockingIOError('Existing GPU compute process '+busy)
        return handles
    except BaseException:
        for f in handles:f.close()
        raise

def freeze(out,cfg):
    paths=[ROOT/'src/research'/x for x in ['deepseek_sboed.py','dad_location_policy.py','policies.py','engine.py','benchmarks.py','compute_budget.py','location_posterior.py','journal_validation.py','location_evaluation.py','moe_v5_conditional.py','moe_v5a_sampler.py','moe_v5a_training.py','moe_v5a_validation.py','moe_v5_experiment.py']]+[ROOT/'src/online_cli.py',ROOT/'run.sh']
    hashes={}
    import shutil
    for p in paths:
        rel=str(p.relative_to(ROOT));dest=out/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[rel]=sha(p)
    write(out/'source_manifest.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sha256=hashes))
    write(out/'configuration.json',cfg);return hashes
def verify_source(hashes):
    bad=[p for p,h in hashes.items() if sha(ROOT/p)!=h]
    if bad:raise RuntimeError('Frozen training source changed: '+','.join(bad))

def draws(benchmark,batch,contrasts,seed,device):
    with torch.device(device):
        g=generator(seed);truth=benchmark.sample((batch,),g);bank=benchmark.sample((batch,contrasts),g)
        noise=torch.stack([torch.randn(batch,1,generator=g,dtype=torch.float64) for _ in range(30)],1)
    return truth,bank,noise

def primary_backward(benchmark,model,truth,bank,noise,seed,microbatch=128):
    """Exact effective-batch objective with global detached balance frequency.

    The cheap truth-only pass fixes global assignment frequencies. Microbatch
    sPCE and balance gradients sum to the original B-wide loss; no optimizer
    update occurs between chunks. This is not an L/B reduction.
    """
    batch=len(truth);contrasts=bank.shape[1];calls=benchmark.calls
    with torch.no_grad():
        first=rollout(benchmark,model,30,batch,seed,0,truth=truth,noise=noise,prior_particles=truth.new_empty(batch,0,4))
        ids=torch.cat([x[1] for x in model._gate_cache]);frequency=torch.nn.functional.one_hot(ids,7).sum(1).double().mean(0)*(7/3)
        del first,ids
    model._gate_cache=[];mean=0.;saturated=0;balance_value=0.
    for start in range(0,batch,microbatch):
        n=min(microbatch,batch-start)
        result=rollout(benchmark,model,30,n,seed,contrasts,truth=truth[start:start+n],noise=noise[start:start+n],prior_particles=bank[start:start+n])
        info=result['info'][:,-1];prob=torch.cat([x[0] for x in model._gate_cache]).mean(0)
        balance=model.balance_factor*(frequency.detach()*prob).sum()
        loss=(-info.mean()+balance)*(n/batch);loss.backward()
        mean+=float(info.detach().sum())/batch;saturated+=int((info.detach()>math.log(contrasts+1)-.1).sum());balance_value+=float(balance.detach())*(n/batch)
        model._gate_cache=[];del result,info,prob,balance,loss
    return dict(main_spce=mean,saturation_fraction=saturated/batch,balance=balance_value,
        main_likelihood_particle_stages=benchmark.calls-calls,extra_truth_only_replay_trajectories=batch)

def auxiliary_backward(benchmark,model,truth,noise,seed,cfg,stage_override=None):
    if cfg['alpha_cond']==0:
        return dict(status='OFF',conditional_score=0.,stage_sum_estimate=0.,alpha=0.,auxiliary_gradient_norm=0.,posterior_proposals=0,total_auxiliary_likelihood_particle_stages=0)
    g=torch.Generator(device=truth.device).manual_seed(seed)
    index=int(torch.randint(0,len(truth),(1,),generator=g,device=truth.device))
    stage=int(torch.randint(0,30,(1,),generator=g,device=truth.device)) if stage_override is None else stage_override
    estimate,stats=conditional_auxiliary(benchmark,model,truth[index:index+1],noise[index:index+1],stage,seed+2000003,cfg['conditional_negatives'],cfg['posterior_particles'],cfg['posterior_moves'])
    gradients=torch.autograd.grad(cfg['alpha_cond']*estimate,tuple(model.parameters()),retain_graph=True,allow_unused=True)
    groups={};total=0.
    for (name,p),grad in zip(model.named_parameters(),gradients):
        if grad is None:continue
        if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite conditional policy gradient')
        norm=float(grad.square().sum());total+=norm
        key='head' if name.startswith('emitter') else '.'.join(name.split('.')[:2]) if name.startswith('routed.') else name.split('.')[0]
        groups[key]=groups.get(key,0.)+norm
    (-cfg['alpha_cond']*estimate).backward();model._gate_cache=[]
    stats.update(alpha=cfg['alpha_cond'],auxiliary_gradient_norm=math.sqrt(total),
        auxiliary_gradient_groups={k:math.sqrt(v) for k,v in groups.items()},
        sampling_distribution_gradient='STOPPED; biased auxiliary gradient, original full-sPCE remains anchor')
    return stats

def utility_module(reference):
    file=reference/'source/src/research/v4_utility_routing.py';spec=importlib.util.spec_from_file_location('v5a_frozen_v4_utility',file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def snapshot(out,model,optimizer,scheduler,update,clock,cfg,hashes,label):
    verify_source(hashes);p=out/'checkpoints'/f'update{update}_{label}.pt';p.parent.mkdir(exist_ok=True)
    tmp=p.with_suffix('.partial');torch.save(dict(method='MoE-sBOED-v5',variant=VARIANT,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),update=update,seed=cfg['seed'],configuration=cfg,config_sha256=sha(out/'configuration.json'),source_sha256=hashes,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],charged_seconds=clock.spent(),initialization='FROM_SCRATCH_NATIVE_V4_SEED'),tmp);tmp.replace(p)
    h=sha(p);mark=clock.checkpoint(str(p));row=dict(path=str(p),sha256=h,update=update,charged_seconds=clock.spent(),eligible=mark['eligible'])
    write(p.with_suffix('.json'),row);return row

def train(args,cfg):
    ref,receipt=verified_reference(ROOT/cfg['reference_bundle']);validation=json.loads((ROOT/cfg['posterior_validation']).read_text())
    if validation['status']!='PASS_EMPIRICAL_APPROXIMATE_SAMPLER':raise ValueError('Posterior validation failed')
    if validation['particles']!=cfg['posterior_particles'] or validation['moves']!=cfg['posterior_moves']:raise ValueError('Sampler settings differ from validated settings')
    smoke=args.action=='smoke';out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    locks=acquire();torch.set_num_threads(4);torch.cuda.init();torch.manual_seed(args.seed)
    effective=copy.deepcopy(cfg);effective['seed']=args.seed
    if smoke:effective.update(batch=32,contrasts=32,microbatch=16,native_max_updates=3,budget_seconds=180,WORKABILITY_ONLY=True)
    if not effective.get('auxiliary_off_control') and effective['budget_seconds']!=14400 and not smoke:raise ValueError('Native four-hour cap required')
    hashes=freeze(out,effective);write(out/'posterior_validation.json',validation)
    resume=None;credit=0.;previous_status=None
    if args.resume_checkpoint:
        oldpath=Path(args.resume_checkpoint).resolve();resume=torch.load(oldpath,map_location='cpu',weights_only=False)
        verify_source(resume['source_sha256'])
        if resume['configuration']!=effective:raise ValueError('Resume configuration differs from frozen training identity')
        oldclock=json.loads((oldpath.parent.parent/'compute_clock.json').read_text())
        if 'stopped_at' not in oldclock:raise ValueError('Unaccounted interrupted time; exact automatic resume is unsafe')
        previous_status=json.loads((oldpath.parent.parent/'run_status.json').read_text())
        pid=previous_status['PID']
        if Path(f'/proc/{pid}').exists():raise RuntimeError('Previous worker still exists; no duplicate resume')
        credit=oldclock['credit_seconds']+oldclock['stopped_at']-oldclock['start']-sum(b-a for a,b,_ in oldclock['excluded'])
        if credit>=effective['budget_seconds']:raise ValueError('No charged allowance remains')
    clock=ComputeClock(out/'compute_clock.json',effective['budget_seconds'],credit_seconds=credit,hardware=torch.cuda.get_device_name(0),start=float(os.environ.get('BOED_V5A_ENTRY_MONOTONIC',ENTRY)))
    p=DeepSeekPolicy().cuda().train();optimizer=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,.98)
    with torch.device('cuda:0'):b=make_benchmark(effective['benchmark'])
    original_utility=utility_module(ref);latest=None;bound=1.;update=0;aux_seconds=0.;next_checkpoint=300.;last={};stopping=[False]
    if resume is not None:
        p.load_state_dict(resume['state_dict'],strict=True);optimizer.load_state_dict(copy.deepcopy(resume['optimizer']));scheduler.load_state_dict(resume['scheduler']);update=resume['update']
        torch.set_rng_state(resume['rng_cpu'].cpu());torch.cuda.set_rng_state_all(resume['rng_cuda'])
        b.calls=previous_status['last_metrics']['actual_particle_stages'];aux_seconds=previous_status['last_metrics']['total_conditional_seconds']
    def stop(signum,frame):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(state,**kw):
        row=dict(state=state,method='MoE-sBOED-v5',variant=VARIANT,seed=args.seed,PID=os.getpid(),update=update,charged_seconds=clock.spent(),primary_training_trajectories=update*effective['batch'],latest_checkpoint=latest,last_metrics=last,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
        write(out/'run_status.json',row)
        if not smoke:write(out.parent/'run_status.json',row)
    if effective.get('auxiliary_off_control') and not smoke:
        latest=snapshot(out,p,optimizer,scheduler,update,clock,effective,hashes,'initial')
    status('RUNNING',gpu=torch.cuda.get_device_name(0),gpu_process_start_verified=True,initialization='FROM_SCRATCH_NO_V4_TRAINED_WEIGHTS')
    try:
        while update<effective['native_max_updates'] and clock.can_start(max(60.,2*bound)) and not stopping[0]:
            started=time.monotonic();attempted_update=update+1;optimizer.zero_grad(set_to_none=True)
            with torch.device('cuda:0'):
                truth,bank,noise=draws(b,effective['batch'],effective['contrasts'],args.seed+attempted_update*104729,'cuda:0')
                main=primary_backward(b,p,truth,bank,noise,args.seed+attempted_update*104729,effective['microbatch'])
                primary_router=p.router.weight.grad.detach().clone();a_start=time.monotonic()
                auxiliary=auxiliary_backward(b,p,truth,noise,61000000+args.seed*100000+attempted_update,effective)
                elapsed=time.monotonic()-a_start;aux_seconds+=elapsed
                old_credit=None
                if attempted_update%100==0:
                    now=p.router.weight.grad.clone();p.router.weight.grad=primary_router.clone()
                    record=original_utility.paired_credit(b,p,39600000+args.seed*100000+attempted_update,64,1024)
                    old_credit=original_utility.add_credit(p,record,.05);delta=p.router.weight.grad-primary_router;p.router.weight.grad=now+delta
                norm=torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True);optimizer.step()
                if (attempted_update-1)%1000==0:scheduler.step()
                torch.cuda.synchronize()
            update=attempted_update;bound=max(bound,time.monotonic()-started)
            last=dict(update=update,**main,conditional=auxiliary,conditional_seconds=elapsed,total_conditional_seconds=aux_seconds,gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],native_v4_utility_credit=old_credit,charged_seconds=clock.spent(),actual_particle_stages=b.calls,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            with (out/'train_seed101.log' if args.seed==101 else out/f'train_seed{args.seed}.log').open('a') as f:f.write(json.dumps(last)+'\n')
            if effective.get('auxiliary_off_control'):
                with (out.parent/'training_progress.csv').open('a',newline='') as pf:
                    pw=csv.writer(pf)
                    if pf.tell()==0:pw.writerow(['update','primary_trajectories','main_spce','saturation_fraction','gradient_norm','charged_seconds','auxiliary'])
                    pw.writerow([update,update*effective['batch'],main['main_spce'],main['saturation_fraction'],float(norm),clock.spent(),'OFF'])
            print(json.dumps(last),flush=True)
            if not all(math.isfinite(x) for x in [main['main_spce'],auxiliary['conditional_score'],float(norm)]):raise FloatingPointError('Nonfinite training state')
            if update==1 or clock.spent()>=next_checkpoint:
                latest=snapshot(out,p,optimizer,scheduler,update,clock,effective,hashes,'recovery');next_checkpoint=clock.spent()+300
            status('RUNNING')
            if smoke and args.smoke_stop_after is not None and update>=args.smoke_stop_after:stopping[0]=True
            del truth,bank,noise
        latest=snapshot(out,p,optimizer,scheduler,update,clock,effective,hashes,'final' if not stopping[0] else 'interrupted')
        if not latest['eligible']:raise RuntimeError('Final checkpoint exceeded charged budget')
        report=clock.stop();status('INTERRUPTED_RECOVERABLE' if stopping[0] else 'INCOMPLETE_FOR_MATCHED_BUDGET' if effective.get('auxiliary_off_control') and update<effective['native_max_updates'] else 'COMPLETE_TRAINING',compute=report,native_schedule_complete=update>=effective['native_max_updates'])
        with (out/'compute_accounting.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['seed','optimizer_updates','fresh_training_trajectories','main_contrasts','actual_simulator_particle_stages','charged_seconds','conditional_seconds','peak_vram_bytes','status']);w.writerow([args.seed,update,update*effective['batch'],effective['contrasts'],b.calls,report['charged_seconds'],aux_seconds,torch.cuda.max_memory_allocated(),'WORKABILITY_ONLY' if smoke else ('MATCHED_UPDATE_TARGET_COMPLETE' if update==effective['native_max_updates'] else 'INCOMPLETE') if effective.get('auxiliary_off_control') else 'COMPUTE_LIMITED' if update<50000 else 'NATIVE_SCHEDULE_COMPLETE'])
        if not smoke and not stopping[0] and (not effective.get('auxiliary_off_control') or update==7714):
            try:evaluate(out,effective,latest)
            except Exception as exc:write(out/'evaluation_pending.json',dict(state='PENDING',error=str(exc),command='See next_commands.sh; training checkpoint is preserved'))
    except BaseException as exc:
        try:
            latest=snapshot(out,p,optimizer,scheduler,update,clock,effective,hashes,'failure_evidence')
        except Exception:pass
        write(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),last_checkpoint=latest,clock=clock.stop()));status('FAILED',error=str(exc));raise
    finally:
        for f in locks:f.close()

def evaluate(out,cfg,checkpoint):
    from src.research.journal_validation import CommonValidation
    files=ROOT/cfg['matched_evaluation_directory'];cases_path=files/'location_cases.npz';bank_path=files/'location_contrasts.npy'
    if sha(cases_path)!=cfg['cases_sha256'] or sha(bank_path)!=cfg['bank_sha256']:raise ValueError('Matched bank identity mismatch')
    with torch.device('cpu'):
        saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False);p=DeepSeekPolicy().eval();p.load_state_dict(saved['state_dict'],strict=True)
    # Existing explicit-likelihood evaluator; checkpoint frozen before these reads.
    v=CommonValidation(cfg['benchmark'],30,1001,2000,1000000,16,16384,device='cuda:0')
    a=np.load(cases_path);v.truth=torch.from_numpy(a['truth']).double();v.noise=torch.from_numpy(a['noise']).double();v.bank=torch.from_numpy(np.load(bank_path)).double();v.identity=hashlib.sha256(v.truth.numpy().tobytes()+v.noise.numpy().tobytes()+v.bank.numpy().tobytes()).hexdigest()
    scores,cost=v(p);np.save(out/f'dev_scores_seed{cfg["seed"]}.npy',scores)
    def stats(x):
        mean=float(x.mean());se=float(x.std(ddof=1)/len(x)**.5);return dict(mean=mean,MC_SE=se,CI95=[mean-1.96*se,mean+1.96*se])
    result=dict(v5a=stats(scores),metadata=cost,checkpoint=checkpoint,evaluation_seed=1001,training_seed=cfg['seed'])
    baseline=np.load(files/f'v4_seed{cfg["seed"]}_scores.npy');result['v5a_minus_v4']=stats(scores-baseline)
    if cfg['seed']==101:
        old=json.loads((files/'vsoed_seed101_scores.json').read_text());key=cfg.get('vsoed_score_key','spce')
        bs=np.array([row[key] for row in old]);result['v5a_minus_native_vsoed']=stats(scores-bs)
    write(out/f'paired_eig_seed{cfg["seed"]}.json',result)
    with (out/f'dev_eig_seed{cfg["seed"]}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','training_seed','eig','MC_SE','evaluation_seed','cases','contrasts']);w.writerow(['V5-A_empirical_SMC',cfg['seed'],result['v5a']['mean'],result['v5a']['MC_SE'],1001,2000,1000000])

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,choices=[101,102,103,104,105],default=101);p.add_argument('--action',choices=['train','smoke','evaluate'],default='train');p.add_argument('--evaluate-checkpoint');p.add_argument('--resume-checkpoint');p.add_argument('--smoke-stop-after',type=int)
    a=p.parse_args(argv);cfg=yaml.safe_load(Path(a.config).read_text())
    global VARIANT
    if cfg.get('auxiliary_off_control'):
        if cfg['alpha_cond']!=0 or cfg.get('v5_b_enabled') or cfg.get('v5_c_enabled') or cfg['native_max_updates']!=7714 or a.seed!=101:raise ValueError('Frozen OFF seed101 matched-work contract required')
        VARIANT='V5-A_auxiliary_OFF_matched7714'
    elif cfg['alpha_cond']!=.1 or cfg.get('v5_b_enabled'):raise ValueError('Frozen V5-A-only configuration required')
    if a.smoke_stop_after is not None and a.action!='smoke':raise ValueError('No artificial update cutoff is allowed in primary training')
    if a.action=='evaluate':
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False);saved=torch.load(a.evaluate_checkpoint,map_location='cpu',weights_only=False);evaluate(out,saved['configuration'],dict(path=a.evaluate_checkpoint,sha256=sha(a.evaluate_checkpoint)));return
    train(a,cfg)

if __name__=='__main__':main()
