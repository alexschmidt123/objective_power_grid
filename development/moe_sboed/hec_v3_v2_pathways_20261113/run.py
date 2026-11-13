"""V2-based four-pathway decision MoE: bounded native-EIG pilot."""
import argparse,copy,gc,json,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from common import ROOT,WORK,n,audit
from pathway_policy import PathwayMoE,Standalone
from native_adapter import build_native_class
from engine import phase,Stop
from resolution import signal,nested_scores,paired

def distill(parent,model,out,steps=300):
    # Use ONLY recorded past designs/observations from parent replay,never theta.
    from soft_policy import prefix_inputs
    rng=np.random.default_rng(202611134101);ids=rng.choice(1000000,4608,replace=False);stages=rng.integers(0,30,4608)
    buffer=parent._runtime_buffer
    x=prefix_inputs(buffer['ds_hist'][torch.tensor(ids,device='cuda')],buffer['ys_hist'][torch.tensor(ids,device='cuda')],torch.tensor(stages,device='cuda'))
    with torch.no_grad():target=parent.actor_net(x)
    np.savez_compressed(out/'DISTILLATION.npz',X=x.cpu().numpy(),target=target.cpu().numpy(),calibration=4096,validation=512,replay_indices=ids,stages=stages)
    receipts={}
    for k,expert in enumerate(model.experts):
        model.experts.requires_grad_(False);expert.head.requires_grad_(True)
        policy=Standalone(model.context,expert,clone=False);opt=torch.optim.Adam(expert.head.parameters(),lr=.001)
        with torch.no_grad():before=float((policy(x[4096:])-target[4096:]).square().mean())
        local=np.random.default_rng(202611134102)
        for step in range(steps):
            idx=torch.tensor(local.integers(4096,size=256),device='cuda');opt.zero_grad(set_to_none=True)
            loss=(policy(x[idx])-target[idx]).square().mean();assert torch.isfinite(loss);loss.backward();opt.step()
        with torch.no_grad():after=float((policy(x[4096:])-target[4096:]).square().mean())
        receipts[f'E{k+1}']=dict(steps=steps,heldout_action_MSE_before=before,heldout_action_MSE_after=after,improved=after<before,
           training_parameters='own decision head only',environment_observations=0)
        for p in expert.parameters():p.grad=None
    model.freeze_experts();torch.save(dict(actor_state=model.state_dict()),out/'INITIAL_DISTILLED.pt')
    n.put(out/'DISTILLATION.json',receipts)
    return receipts

def optimizer(a,actor):
    lr=a.actor_optimizer.param_groups[0]['lr'];a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor)
    a.actor_optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=lr)
    a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999)

@torch.no_grad()
def rollout(actor,lib,truth,noise,u,mode='learned',expert=None):
    ds=torch.zeros(len(truth),30,2,device='cuda');ys=torch.zeros(len(truth),30,device='cuda',dtype=torch.float64)
    counts=torch.zeros(4,device='cuda',dtype=torch.long)
    for t in range(30):
        x=lib.packed(ds,ys,t)
        if isinstance(actor,PathwayMoE):
            if mode=='expert':action=actor(x,mode=mode,expert=expert);counts[expert]+=len(x)
            else:
                action,info=actor(x,mode=mode,uniforms=u[:,t],collect=True);counts+=torch.bincount(info['selected'],minlength=4)
        else:action=actor(x)
        assert torch.isfinite(action).all() and bool((action.abs()<=4.000001).all())
        ds[:,t]=action;ys[:,t]=signal(truth.reshape(-1,2,2),action.double())+.5*noise[:,t]
    return ds,ys,counts

def evaluate(out,cfg,deadline):
    lib=n.load_library();parent=lib.load_frozen('cuda');model=PathwayMoE(parent).cuda()
    model.load_state_dict(torch.load(out/'selector/actor.pt',map_location='cuda',weights_only=True)['actor_state']);model.eval()
    rng=np.random.default_rng(cfg['case_seed']);truth=rng.standard_normal((cfg['cases'],4));noise=rng.standard_normal((cfg['cases'],30))
    uniforms=np.random.default_rng(cfg['routing_seed']).random((cfg['cases'],30))
    np.savez_compressed(out/'EVALUATION_CASES.npz',truth=truth,noise=noise,uniforms=uniforms)
    theta=torch.tensor(truth,device='cuda');eps=torch.tensor(noise,device='cuda')
    u=torch.tensor(uniforms,device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
    bank=np.random.default_rng(cfg['contrast_seed']).standard_normal((cfg['contrasts'],4));contrasts=torch.tensor(bank,device='cuda');del bank
    scores={};counts={}
    for mode in ['original','control','learned','uniform','expert1','expert2','expert3','expert4','vsoed']:
        current=model
        if mode in ['original','control']:current=parent
        if mode=='control':parent.load_state_dict(torch.load(out/'control/actor.pt',map_location='cuda',weights_only=True)['actor_state'])
        if mode=='vsoed':
            import evaluate_v2
            current,path=evaluate_v2.actor_load('vsoed')
            manifest=json.loads((audit.BASE/'ARTIFACT_SHA256.json').read_text())['files'];assert n.sha(path)==manifest['inputs/vsoed.pt']
        values=[];selected=torch.zeros(4,device='cuda',dtype=torch.long)
        for first in range(0,cfg['cases'],16):
            if time.monotonic()>deadline:raise Stop('Evaluation cap')
            ds,ys,c=rollout(current,lib,theta[first:first+16],eps[first:first+16],u[first:first+16],
                           mode='expert' if mode.startswith('expert') else 'uniform' if mode=='uniform' else 'learned',
                           expert=int(mode[-1])-1 if mode.startswith('expert') else None)
            values.append(nested_scores(ds,ys,theta[first:first+16],contrasts,[cfg['contrasts']],deadline=deadline)[cfg['contrasts']].cpu().numpy());selected+=c
            n.put(out/'STATUS.json',dict(status='EVALUATING',method=mode,completed=min(first+16,cfg['cases']),total=cfg['cases']))
        scores[mode]=np.concatenate(values);counts[mode]=selected.cpu().tolist();np.save(out/f'{mode}_scores.npy',scores[mode])
        if mode=='vsoed':del current;gc.collect();torch.cuda.empty_cache()
    del contrasts,theta,eps,u,parent,model;gc.collect();torch.cuda.empty_cache()
    for method in ['dad','rl_sboed','scorebed']:
        if time.monotonic()>deadline:raise Stop('Historical evaluation cap')
        n.put(out/'STATUS.json',dict(status='EVALUATING',method=method))
        subprocess.run([sys.executable,'-u',str(pathlib.Path(__file__).parent/'baseline_worker.py'),'--method',method,'--output',str(out)],check=True,timeout=max(1,deadline-time.monotonic()))
        scores[method]=np.load(out/f'{method}_scores.npy')
    comparisons={method:paired(scores['learned'],score) for method,score in scores.items() if method not in ['learned','control']}
    maximum=0
    for i,key in enumerate(sorted(comparisons,key=lambda key:comparisons[key]['p_two_sided'])):
        maximum=max(maximum,min(1,(len(comparisons)-i)*comparisons[key]['p_two_sided']));comparisons[key]['p_Holm']=maximum
    primary=paired(scores['learned'],scores['control'])
    accepted=primary['CI95'][0]>0 and all(v['mean']>0 and v['p_Holm']<.05 for v in comparisons.values())
    result=dict(status='COMPLETE',accepted=accepted,primary_learned_minus_control=primary,secondary_comparisons=comparisons,
       means={k:float(v.mean()) for k,v in scores.items()},selection_counts=counts,
       limitations='Short pilot,one training lineage,finite prior-sPCE;historical DAD/vsOED/RL-sBOED/ScoreBED not compute matched. Expert query mechanisms differ by construction,but useful conditional specialization and baseline superiority require measured results.')
    n.put(out/'RESULTS.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(experts=4,distillation_steps=300,expert_updates=250,selector_updates=250,control_updates=1250,
       wall_seconds=14400,cases=1024,contrasts=16000000,case_seed=202611134201,routing_seed=202611134202,contrast_seed=202611134203,
       context='original frozen V2 history/commander/belief/base',expert_training='own V2 query core plus own action head;new Adam at parent terminalLR',
       router='inherited V2 coordinator+response_gate,zero-init8->4 proposal term;categorical expected nativeQ',
       baselines=['uniform','expert1','expert2','expert3','expert4','original V2','matched V2','DAD','vsOED','RL-sBOED','ScoreBED'],
       source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    n.put(out/'CONFIG.json',cfg)
    paths=[*sorted(pathlib.Path(__file__).parent.glob('*.*')),*sorted((audit.BASE/'source').glob('*.py')),
           *sorted((n.P/'source').rglob('*.py')),audit.BASE/'arms/candidate/actor.pt',audit.BASE/'arms/candidate/runtime.pkl',
           audit.BASE/'inputs/HISTORICAL_BASELINES.json',audit.BASE/'inputs/vsoed.pt']
    protected={str(f):n.sha(f) for f in paths if f.is_file()};n.put(out/'PRE_RUN_MANIFEST.json',protected)
    n.original.setup()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds']
    try:
        state=n.load_parent();a=state['algorithm'];model=PathwayMoE(a.actor_net).cuda();receipts=distill(a,model,out)
        if not all(v['improved'] for v in receipts.values()):raise Stop('Head distillation failed heldout action-competence prerequisite;no long training')
        initial=copy.deepcopy(model).cpu();del a,state,model;gc.collect();torch.cuda.empty_cache()
        for k in range(4):
            state=n.load_parent();a=state['algorithm'];single=initial.standalone(k).cuda();single.expert.requires_grad_(True)
            optimizer(a,single);n.original.restore(state['rng'])
            result=phase(a,f'expert{k+1}',cfg['expert_updates'],out,start,cfg['wall_seconds'],seed=202611134300+k)
            if result['status']!='COMPLETE':raise Stop(result['status'])
            del a,state,single;gc.collect();torch.cuda.empty_cache()
        state=n.load_parent();a=state['algorithm'];model=copy.deepcopy(initial).cuda();target=copy.deepcopy(model)
        for k in range(4):
            cp=torch.load(out/f'expert{k+1}/actor.pt',map_location='cuda',weights_only=True)
            model.experts[k].load_state_dict({key[7:]:v for key,v in cp['actor_state'].items() if key.startswith('expert.')})
            target.experts[k].load_state_dict({key[7:]:v for key,v in cp['actor_target_state'].items() if key.startswith('expert.')})
        model.freeze_experts();target.freeze_experts();cls,patch=build_native_class(type(a));a.__class__=cls
        optimizer(a,model);a.actor_target_net=target;n.put(out/'NATIVE_ACTOR_PATCH.json',patch)
        before=n.digest({k:v for k,v in model.state_dict().items() if k.startswith(('context.','experts.'))})
        n.original.restore(state['rng']);result=phase(a,'selector',cfg['selector_updates'],out,start,cfg['wall_seconds'],seed=202611134310)
        assert before==n.digest({k:v for k,v in model.state_dict().items() if k.startswith(('context.','experts.'))})
        if result['status']!='COMPLETE':raise Stop(result['status'])
        del a,state,model,target;gc.collect();torch.cuda.empty_cache()
        state=n.load_parent();a=state['algorithm'];n.original.restore(state['rng'])
        result=phase(a,'control',cfg['control_updates'],out,start,cfg['wall_seconds'])
        if result['status']!='COMPLETE':raise Stop(result['status'])
        del a,state;gc.collect();torch.cuda.empty_cache()
        result=evaluate(out,cfg,deadline)
        assert all(n.sha(f)==v for f,v in protected.items())
        n.put(out/'STATUS.json',dict(status='COMPLETE',accepted=result['accepted'],historical_artifacts_preserved=True,seconds=time.monotonic()-start))
    except Stop as exc:n.put(out/'STATUS.json',dict(status='INCOMPLETE',reason=str(exc)))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
