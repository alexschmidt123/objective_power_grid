"""Fresh three-split V2 competency and consumed-feature sensitivity audit."""
import argparse,hashlib,importlib.util,json,os,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
PARENT=WORK/'development/moe_sboed/hec_v2_router_only_20261112'
spec=importlib.util.spec_from_file_location('native_v2_restore',PARENT/'train_router.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
from v2_tasks import labels
from resolution import paired
BASE=native.BASE;put=native.put;sha=native.sha

def representation(slots):
    """All128 packet coordinates; exclude ONLY actual analytic E2/E4 prefix."""
    z=slots.clone();z[:,[1,3],64:76]=0
    return z

def ridge_fit(z,y):
    mu=z.mean(0);sd=z.std(0).clip(1e-5);x=np.column_stack([np.ones(len(z)),(z-mu)/sd])
    beta=np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@y)
    return mu,sd,beta

def ridge_predict(model,z):
    mu,sd,beta=model
    return np.column_stack([np.ones(len(z)),(z-mu)/sd])@beta

def grouped_mean(x,n):return np.asarray(x).reshape(n,3).mean(1)

def analyze(z,history,targets,readouts,counts):
    n=sum(counts);edges=np.cumsum([0,*counts]);fit=slice(0,3*counts[0]);validation=slice(3*counts[0],3*(counts[0]+counts[1]));test=slice(validation.stop,3*n)
    losses=np.empty((5,4,len(z)));models={};constant={};normalization={}
    for task,y in enumerate(targets):
        scale=y[fit].std(0).clip(.1);normalization[task]=scale
        constant[task]=np.square((y-y[fit].mean(0))/scale).mean(-1)
        for expert in range(5):
            x=z[:,expert] if expert<4 else history
            model=ridge_fit(x[fit],y[fit]);models[expert,task]=model
            losses[expert,task]=np.square((ridge_predict(model,x)-y)/scale).mean(-1)
    result={};readout_stats={}
    for k in range(4):
        rivals=[j for j in range(4) if j!=k]
        rival=min(rivals,key=lambda j:losses[j,k,validation].mean())
        testloss=grouped_mean(losses[k,k,test],counts[2]);base=grouped_mean(constant[k][test],counts[2])
        result[f'E{k+1}']=dict(target_dimensions=targets[k].shape[1],
             selected_rival=f'E{rival+1}',selection_uses_validation_only=True,
             designated_advantage=paired(grouped_mean(losses[rival,k,test],counts[2]),testloss),
             decoded_vs_constant=paired(base,testloss),
             decoded_vs_history=paired(grouped_mean(losses[4,k,test],counts[2]),testloss),
             normalized_MSE=float(testloss.mean()),constant_normalized_MSE=float(base.mean()),
             relative_MSE_reduction=float(1-testloss.mean()/base.mean()))
    for key,k in [('E2',1),('E4',3)]:
        scale=normalization[k];oldloss=np.square((readouts[key]-targets[k])/scale).mean(-1)
        probe=losses[k,k]
        readout_stats[key]=dict(existing_readout_vs_constant=paired(grouped_mean(constant[k][test],counts[2]),grouped_mean(oldloss[test],counts[2])),
                               recalibrated_probe_vs_existing=paired(grouped_mean(oldloss[test],counts[2]),grouped_mean(probe[test],counts[2])))
    # Multiple designated-advantage tests; probes don't alter actor parameters.
    ordered=sorted(result,key=lambda k:result[k]['designated_advantage']['p_two_sided']);pmax=0
    for rank,key in enumerate(ordered):
        pmax=max(pmax,min(1,(4-rank)*result[key]['designated_advantage']['p_two_sided']))
        result[key]['designated_advantage']['p_Holm4']=pmax
    return dict(experts=result,existing_diagnostic_readouts=readout_stats,
                feature_dimensions=128,calibration_episodes=counts[0],selection_episodes=counts[1],test_episodes=counts[2]),losses

def sensitivity(actor,x):
    """Local direct-fusion derivative; gates held fixed, no router derivative."""
    with torch.no_grad():_,info=actor(x,collect=True);a0=actor.base(x)
    slots=info['slots'].detach().requires_grad_(True);g=info['gates'].detach()
    pre=actor.fusion[0](torch.cat([info['history'].detach(),(g[...,None]*slots[...,:64]).flatten(1)],-1))
    pre=pre+actor.response_fusion((g[...,None]*slots[...,64:]).flatten(1))
    delta=actor.fusion[2](actor.fusion[1](pre)).tanh();action=a0+4*delta-a0*delta.abs()
    grads=[torch.autograd.grad(action[:,j].sum(),slots,retain_graph=j==0)[0] for j in range(2)]
    jac=torch.stack(grads,1)
    assert torch.isfinite(jac).all()
    return jac.square().sum((1,3)).sqrt().detach().cpu().numpy()

def tests():
    x=torch.arange(2*4*128,dtype=torch.float32).reshape(2,4,128)
    y=representation(x)
    assert torch.equal(x[:,[0,2]],y[:,[0,2]])
    assert bool((y[:,[1,3],64:76]==0).all())
    assert torch.equal(x[:,[1,3],76:],y[:,[1,3],76:])
    rng=np.random.default_rng(91);z=rng.standard_normal((256,4));target=z[:,:2]*2
    model=ridge_fit(z[:128],target[:128]);prediction=ridge_predict(model,z[128:])
    assert np.mean((prediction-target[128:])**2)<.002

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--run',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(smoke=args.smoke,episodes=[8,4,8] if args.smoke else [256,128,256],stages=[3,14,26],
             teacher_replicates=2 if args.smoke else 16,seed=202611132701,wall_seconds=300 if args.smoke else 1800,
             target_definition='Unmodified frozen v2_tasks.labels; E1 geometry14,E2 rotated discrimination3,E3 sensitivity9,E4 second-step information/tail6. Model-derived, approximate-belief targets; not exact EIG.',
             source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    put(out/'CONFIG.json',cfg)
    files=[*sorted(pathlib.Path(__file__).parent.glob('*.py')),*sorted((BASE/'source').glob('*.py')),
           PARENT/'train_router.py',BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl']
    manifest={str(f):sha(f) for f in files};put(out/'PRE_RUN_MANIFEST.json',manifest)
    native.original.setup();tests()
    while True:
        try:lease=native.original.acquire();break
        except BlockingIOError:put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds']
    try:
        state=native.load_parent();a=state['algorithm'];actor=a.actor_net.eval().requires_grad_(False)
        before=native.digest(actor.state_dict());lib=native.load_library();n=sum(cfg['episodes']);rng=np.random.default_rng(cfg['seed'])
        theta=torch.tensor(rng.standard_normal((n,4)),device='cuda');eps=torch.tensor(rng.standard_normal((n,30)),device='cuda')
        ds,ys,_=lib.rollout(actor,theta,eps,deadline=deadline);del theta,eps
        x=torch.cat([lib.packed(ds,ys,t)[:,None] for t in cfg['stages']],1).reshape(-1,117)
        np.savez_compressed(out/'FRESH_HISTORIES.npz',X=x.cpu().numpy(),split_episode_counts=cfg['episodes'])
        z=[];history=[];targets=[[] for _ in range(4)];readouts={'E2':[],'E4':[]};calls=0
        gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+1)
        for first in range(0,len(x),64):
            if time.monotonic()>deadline:raise TimeoutError('Fixed audit budget')
            xx=x[first:first+64]
            with torch.no_grad(),torch.device('cuda'):
                _,info=actor(xx,collect=True)
                target,receipt=labels(a,actor,actor,xx,info,gen,cfg['teacher_replicates'])
                calls+=receipt['predictive_observations']
                z.append(representation(info['slots']).cpu().numpy());history.append(info['history'].cpu().numpy())
                readouts['E2'].append(actor.response_readouts[0](info['responses'][:,1,12:]).cpu().numpy())
                readouts['E4'].append(actor.response_readouts[1](info['responses'][:,3,12:]).cpu().numpy())
                for k in range(4):targets[k].append(target[k].cpu().numpy())
            put(out/'STATUS.json',dict(status='LABELING',completed=min(first+64,len(x)),total=len(x),seconds=time.monotonic()-start))
        z=np.concatenate(z).astype(float);history=np.concatenate(history).astype(float)
        targets=[np.concatenate(y).astype(float) for y in targets];readouts={k:np.concatenate(v).astype(float) for k,v in readouts.items()}
        result,losses=analyze(z,history,targets,readouts,cfg['episodes'])
        jac=sensitivity(actor,x[-min(192,len(x)):])
        result['direct_fusion_action_Jacobian']=dict(mean_norm_by_expert=jac.mean(0).tolist(),
              fraction_exact_zero_by_expert=(jac==0).mean(0).tolist(),router_derivative_excluded=True,
              interpretation='Local sensitivity only; not evidence of useful decision specialization.')
        result['source_findings']=dict(legacy_probe_drops_E1_E3_learned_response_coordinates=True,
             fixed_probe_preserves_all_E1_E3_coordinates=True,
             response_readouts_used_by_deployed_action=False,
             scope='Confirmed diagnostic construction issue; no actor computation bug established. Existing readout mismatch alone does not authorize policy retraining.')
        result.update(status='COMPLETE',smoke=args.smoke,seconds=time.monotonic()-start,
             teacher_predictive_observations=calls,history_simulator_observations=n*30,optimizer_steps=0,
             limitations='One checkpoint. Probes assess decodability of existing model-derived targets, not EIG or scientific accuracy. E4 labels use current V2 posterior/critic and16 conditional replicates; comparison with historical V1-teacher diagnostics is not numerically matched. Diagnostic readouts are not in the action graph.')
        np.savez_compressed(out/'RAW_COMPETENCY.npz',features=z,history=history,losses=losses,jacobian_norm=jac,**{f'target{k}':y for k,y in enumerate(targets)})
        assert native.digest(actor.state_dict())==before
        assert all(sha(f)==v for f,v in manifest.items())
        put(out/'RESULTS.json',result);put(out/'STATUS.json',dict(status='COMPLETE',seconds=time.monotonic()-start,historical_artifacts_preserved=True,optimizer_steps=0))
    except Exception as exc:put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()

if __name__=='__main__':main()
