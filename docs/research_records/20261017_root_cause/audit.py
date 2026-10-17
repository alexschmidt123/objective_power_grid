"""CPU-only saved-artifact instability audit; no training or checkpoint selection."""
import os,sys,json,hashlib,time,math,copy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
if hasattr(os,'sched_getaffinity'):
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[-2:])
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid')
sys.path.insert(0,str(R))
import numpy as np,torch
torch.set_num_threads(2)
from src.research.benchmarks import make_benchmark
from src.research.moe_family import load_policy
from src.research.policies import history_features
OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget'
C=R/'experiments/10162026/10162026_location_root_cause_v3'
C.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),p50=float(np.median(x)),p90=float(np.quantile(x,.9)),p99=float(np.quantile(x,.99)),maximum=float(x.max())) if len(x) else None
def trace_records(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)
def summarize_training(run,native):
    t=read(run/'models'/f'{native}_training.json')
    paths=list((run/'models').glob('*diagnostics.jsonl'))
    rows=list(trace_records(paths[0])) if paths else t['history']
    phases=[]
    for lo,hi in [(0,.3),(.3,.6),(.6,.8),(.8,.95),(.95,1.000001)]:
        rr=[x for x in rows if lo*1e6<=x.get('primary_episodes',0)<hi*1e6]
        result=dict(fraction=[lo,hi],updates=len(rr))
        if rr:
            result['training_spce']=stats([x['training_eig'] for x in rr])
            result['gamma']=stats([x['gamma'] for x in rr])
            norms=[x['gradient_before']['router'] for x in rr]
            result.update(router_gradient=stats(norms),router_zero_fraction=float(np.mean(np.asarray(norms)<=1e-12)),router_negligible_fraction=float(np.mean(np.asarray(norms)<=1e-8)))
            result['experts']={}
            for j in range(4):
                key='expert_'+str(j);v=np.array([x['gradient_before'].get(key,0) for x in rr])
                result['experts']['E'+str(j+1)]=dict(non_negligible_gradient_frequency=float(np.mean(v>1e-8)),gradient_norm=stats(v),
                    mean_training_argmax_selection=float(np.mean([x.get('hard_choice_frequency',[float('nan')]*4)[j] for x in rr])) if 'hard_choice_frequency' in rr[0] else None)
        phases.append(result)
    window=200 if paths else 5
    smooth=[(rows[k+window-1]['primary_episodes'],float(np.mean([x['training_eig'] for x in rows[k:k+window]]))) for k in range(0,len(rows)-window+1,window)]
    best=-1e9;onset=None
    for k,(episodes,mean) in enumerate(smooth):
        if episodes>=300000 and k+2<len(smooth) and all(z[1]<best-.3 for z in smooth[k:k+3]):onset=dict(episodes=episodes,window_mean=mean,previous_best=best,threshold=.3,consecutive_windows=3);break
        best=max(best,mean)
    validations=[]
    vp=run/'models'/f'{native}_validation.jsonl'
    if vp.exists():validations=list(trace_records(vp))
    if not validations:validations=t.get('journal_validation_history',t.get('validation_history',[]))
    input_hashes={str(p):sha(p) for p in [run/'models'/f'{native}.pt',run/'models'/f'{native}_training.json',*paths]}
    return dict(generated_episodes=t.get('generated_training_trajectories'),checkpoint_selection=t.get('checkpoint_selection'),
        trace_resolution='every_update' if paths else 'logged_updates_only',phases=phases,training_smoothed_curve=smooth,
        operational_training_deterioration_onset=onset,validation_history=validations,input_sha256=input_hashes,
        historical_training_logits_saved=False,
        missing_measurement='Raw selector logits were not logged historically; final checkpoint diagnostics cannot reconstruct historical saturation trajectories.')

def policy_diagnostic(run,native,rows,reference_rows):
    b=make_benchmark(read(run/'run_config.json')['benchmark']);h=30
    p=load_policy(run/'models'/f'{native}.pt',b).cpu().eval().requires_grad_(False)
    result={}
    for source,rr in [('own_policy_history',rows[:128]),('common_failed103_history',reference_rows[:128])]:
        actions=torch.tensor([x['actions'] for x in rr],dtype=torch.float64).transpose(0,1)
        observations=torch.tensor([x['observations'] for x in rr],dtype=torch.float64).transpose(0,1)
        n=len(rr);stage_rows=[]
        with torch.no_grad():
            p.set_routing('learned_switch',False)
            for stage in range(h):
                f=history_features(b,list(actions[:stage]),list(observations[:stage]),n,h)
                action=p(f,stage)
                if stage==0:continue
                context=p.context(f);logits=p.router.bits(context.mean(1)).detach().reshape(-1,2)
                scaled=logits/p.router.gamma
                derivative=torch.where(abs(scaled)<.5,(1.5-6*scaled.square())/p.router.gamma,torch.zeros_like(scaled))
                candidates=torch.stack([b.feasible(expert(context[:,j]),list(actions[:stage]),h) for j,expert in enumerate(p.experts)],1)
                pairs=[torch.linalg.vector_norm(candidates[:,j]-candidates[:,k],dim=-1).mean().item() for j in range(4) for k in range(j+1,4)]
                stage_rows.append(dict(stage=stage+1,logit_quantiles=np.quantile(logits.numpy(),[0,.01,.1,.5,.9,.99,1]).tolist(),
                    saturated_bit_fraction=float((logits.abs()>=p.router.gamma/2).double().mean()),
                    fully_saturated_selector_fraction=float((logits.abs()>=p.router.gamma/2).all(-1).double().mean()),
                    selector_derivative=stats(derivative.numpy().ravel()),context_norm=stats(context[:,0].norm(dim=-1).numpy()),
                    mean_pairwise_bounded_expert_action_distance=float(np.mean(pairs)),
                    hard_choices=torch.bincount(p.last_route['weights'].argmax(-1),minlength=4).double().div(n).tolist()))
        # Instantaneous sensitivity only: previous relaxed strategy state is held
        # fixed. This is not a new rollout EIG estimator or a causal intervention.
        sensitivities={}
        for relaxed in [False,True]:
            p.set_routing('learned_switch',relaxed)
            with torch.no_grad():
                for stage in range(h-1):
                    f=history_features(b,list(actions[:stage]),list(observations[:stage]),n,h);p(f,stage)
            f=history_features(b,list(actions[:h-1]),list(observations[:h-1]),n,h).detach().requires_grad_()
            action=b.feasible(p(f,h-1),list(actions[:h-1]),h)
            jj=[]
            for j in range(2):
                grad=torch.autograd.grad(action[:,j].sum(),f,retain_graph=j==0)[0]
                jj.append(grad[:,3::4]/b.obs_scale)
            sensitivity=torch.stack(jj,1).square().sum((1,2)).sqrt()
            sensitivities['relaxed' if relaxed else 'hard']=stats(sensitivity.detach().numpy())
        result[source]=dict(stage_statistics=stage_rows,instantaneous_action_observation_jacobian_norm=sensitivities,
            mean_saturated_bit_fraction=float(np.mean([x['saturated_bit_fraction'] for x in stage_rows])),
            mean_fully_saturated_selector_fraction=float(np.mean([x['fully_saturated_selector_fraction'] for x in stage_rows])))
    return result

def main():
    reg=[e for e in read(OLD/'registry.json') if e.get('benchmark')=='location' and e['method'] in {'moe','switchmoe','scorebed'}]
    refs={}
    for method in ['moe','switchmoe']:
        e=next(e for e in reg if e['method']==method and e['seed']==103)
        refs[method]=read(Path(e['run'])/'eval'/f"{e.get('native_method',method)}_seed1001.json")
    output=[]
    for e in reg:
        method=e['method'];native=e.get('native_method',method);run=Path(e['run'])
        if method=='scorebed':continue
        rows=read(run/'eval'/f'{native}_seed1001.json')
        x=np.asarray([v['spce'] for v in rows])
        entry=dict(method=method,seed=e['seed'],run=str(run),eig=float(x.mean()),training=summarize_training(run,native),
                   final_selector_diagnostics=policy_diagnostic(run,native,rows,refs[method]))
        output.append(entry);write(C/'moe_evidence.json',output)
        print('AUDITED',method,e['seed'],'EIG',entry['eig'],flush=True)
    write(C/'audit_complete.json',dict(status='complete',models=len(output),training_runs_launched=0,
        hardware='CPU only, 2-core affinity',reserved_final_seed_used=False))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback;write(C/'audit_failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
