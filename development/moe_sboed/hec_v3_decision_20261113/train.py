"""Bounded V3 native-EIG pilot; independent complete experts, then selector."""
import argparse,copy,gc,json,pathlib,random,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from common import ROOT,WORK,n,audit
from policy import DecisionMoE
from native_adapter import build_native_class
from resolution import signal

class Stop(Exception):pass
class Table:
    def __init__(self):self.data={}
    def record(self,k,v):self.data[k]=v
    def clear(self):self.data.clear()

def all_optimizers(a):
    return [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(o,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for o in oo]

class Logger:
    def __init__(self,a,phase,out,cfg,start):
        self.algorithm=a;self.phase=phase;self.out=out;self.cfg=cfg;self.start=start
        self.base=int(a.update);self.logger=self;self.tabular=Table();self.counts=dict(actor=0,critic=0,posterior=0)
    def push_prefix(self,*unused):pass
    def pop_prefix(self,*unused):pass
    def log(self,*unused):pass
    def dump_all(self):
        u=int(self.algorithm.update)-self.base
        assert self.counts==dict(actor=u,critic=5*u,posterior=150*u)
        if u==1 or u%25==0 or u==self.cfg['updates']:
            row=dict(status='TRAINING',phase=self.phase,updates=u,target=self.cfg['updates'],
                     optimizer_steps=self.counts,new_episodes=1000*u,new_observations=30000*u,seconds=time.monotonic()-self.start)
            n.put(self.out/'STATUS.json',row);print(json.dumps(row),flush=True)
        if (self.out/'STOP_REQUESTED').exists() or time.monotonic()-self.start>self.cfg['wall_seconds']:raise Stop('Bounded stop')

def phase(a,name,updates,out,start,wall_seconds,seed=None):
    root=out/name;root.mkdir();cfg=dict(updates=updates,wall_seconds=wall_seconds)
    logger=Logger(a,name,out,cfg,start);a.dowel=logger;a.post_approx.dowel=logger;a.save_path=None
    for opt,key in all_optimizers(a):
        opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
        def count(*unused,key=key):logger.counts[key]+=1
        opt.register_step_post_hook(count)
    def monitor(*unused):
        for key,p in a.actor_net.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),key
            if not p.requires_grad:assert p.grad is None,key
    a.actor_optimizer.register_step_pre_hook(monitor)
    if seed is not None:
        torch.manual_seed(seed);np.random.seed(seed%(2**32));random.seed(seed)
    before_buffers=n.digest(dict(a.actor_net.named_buffers()))
    initial=n.digest(a.actor_net.state_dict());base=int(a.update);status='COMPLETE'
    n.put(root/'STARTED.json',dict(phase=name,parent_update=base,initial_actor_digest=initial,seed=seed,
                                  trainable_parameters=sum(p.numel() for p in a.actor_net.parameters() if p.requires_grad)))
    profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
    try:n.original.train_native(a,logger,profile,target=base+updates)
    except Stop:status='INCOMPLETE'
    assert n.digest(dict(a.actor_net.named_buffers()))==before_buffers
    for opt,_ in all_optimizers(a):opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
    torch.save(dict(actor_state=a.actor_net.state_dict(),actor_target_state=a.actor_target_net.state_dict(),native_iteration=int(a.update)),root/'actor.pt')
    with (root/'runtime.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng(),phase=name),f,protocol=5)
    result=dict(status=status,phase=name,updates=int(a.update)-base,optimizer_steps=logger.counts,
       actor_sha256=n.sha(root/'actor.pt'),runtime_sha256=n.sha(root/'runtime.pkl'),frozen_buffers_preserved=True)
    n.put(root/'RESULT.json',result)
    return result

def migrate_selector(a,out):
    actor=DecisionMoE(a.actor_net,experts=3).cuda();target=copy.deepcopy(actor)
    for k in range(3):
        saved=torch.load(out/f'expert{k+1}/actor.pt',map_location='cuda',weights_only=True)
        actor.experts[k].core.load_state_dict(saved['actor_state'])
        target.experts[k].core.load_state_dict(saved['actor_target_state'])
    actor.set_phase('router');target.set_phase('router')
    frozen_before=n.digest({k:v for k,v in actor.state_dict().items() if k.startswith('experts.')})
    cls,patch=build_native_class(type(a));a.__class__=cls
    lr=a.actor_optimizer.param_groups[0]['lr'];a.actor_net=actor;a.actor_target_net=target
    a.actor_optimizer=torch.optim.Adam(actor.router.parameters(),lr=lr)
    a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999)
    n.put(out/'NATIVE_ACTOR_PATCH.json',patch)
    return frozen_before

@torch.no_grad()
def rollout(actor,lib,theta,noise,uniforms,mode,expert=None):
    ds=torch.zeros(len(theta),30,2,device='cuda');ys=torch.zeros(len(theta),30,device='cuda',dtype=torch.float64)
    counts=torch.zeros(3,device='cuda',dtype=torch.long)
    for t in range(30):
        x=lib.packed(ds,ys,t)
        if isinstance(actor,DecisionMoE):
            action,info=actor(x,mode=mode,expert=expert,uniforms=uniforms[:,t],collect=True)
            counts+=torch.bincount(info['selected_expert'].flatten(),minlength=3)
        else:action=actor(x)
        assert bool((action.abs()<=4.000001).all()) and torch.isfinite(action).all()
        ds[:,t]=action;ys[:,t]=signal(theta.reshape(-1,2,2),action.double())+.5*noise[:,t]
    return ds,ys,counts

def evaluate(out,deadline,cases):
    from resolution import nested_scores,paired
    lib=n.load_library();parent=lib.load_frozen('cuda');v3=DecisionMoE(parent).cuda()
    v3.load_state_dict(torch.load(out/'selector/actor.pt',map_location='cuda',weights_only=True)['actor_state']);v3.eval()
    rng=np.random.default_rng(202611133201)
    theta=torch.tensor(rng.standard_normal((cases,4)),device='cuda');noise=torch.tensor(rng.standard_normal((cases,30)),device='cuda')
    uniforms=torch.tensor(np.random.default_rng(202611133202).random((cases,30)),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
    np.savez_compressed(out/'EVALUATION_CASES.npz',truth=theta.cpu().numpy(),noise=noise.cpu().numpy(),uniforms=uniforms.cpu().numpy())
    array=np.random.default_rng(202611133203).standard_normal((16000000,4))
    import hashlib
    n.put(out/'EVALUATION_BANK.json',dict(seed=202611133203,sha256=hashlib.sha256(array.view(np.uint8)).hexdigest()))
    contrasts=torch.tensor(array,device='cuda');del array
    scores={};selection={}
    for mode in ['original','control','learned','uniform','expert1','expert2','expert3']:
        if mode=='control':parent.load_state_dict(torch.load(out/'control/actor.pt',map_location='cuda',weights_only=True)['actor_state'])
        model=parent if mode in ['original','control'] else v3
        values=[];counts=torch.zeros(3,device='cuda',dtype=torch.long);saved_actions=[];saved_observations=[]
        for first in range(0,cases,16):
            if time.monotonic()>deadline:raise Stop('Evaluation wall cap')
            th,eps,u=theta[first:first+16],noise[first:first+16],uniforms[first:first+16]
            name='expert' if mode.startswith('expert') else mode
            action,obs,c=rollout(model,lib,th,eps,u,name,int(mode[-1])-1 if name=='expert' else None)
            if mode=='learned':saved_actions.append(action.cpu().numpy());saved_observations.append(obs.cpu().numpy())
            values.append(nested_scores(action,obs,th,contrasts,[16000000],deadline=deadline)[16000000].cpu().numpy());counts+=c
            n.put(out/'STATUS.json',dict(status='EVALUATING',mode=mode,completed=min(first+16,cases),total=cases))
        scores[mode]=np.concatenate(values);selection[mode]=counts.cpu().tolist();np.save(out/f'{mode}_scores.npy',scores[mode])
        if mode=='learned':np.savez_compressed(out/'LEARNED_HISTORIES.npz',actions=np.concatenate(saved_actions),observations=np.concatenate(saved_observations))
    comparisons={k:paired(scores['learned'],scores[k]) for k in ['uniform','expert1','expert2','expert3']}
    previous=0
    for i,k in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['p_two_sided'])):
        previous=max(previous,min(1,(4-i)*comparisons[k]['p_two_sided']));comparisons[k]['p_Holm4']=previous
    result=dict(primary_learned_minus_control=paired(scores['learned'],scores['control']),
                secondary_learned_minus_original=paired(scores['learned'],scores['original']),
                internal_comparisons=comparisons,means={k:float(v.mean()) for k,v in scores.items()},selection_counts=selection,
                limitations='Short pilot,one lineage,finite16M prior-sPCE,episode/simulator budget matched but compute/parameters differ. Independent native-EIG training does not guarantee expert specialization or named strategy roles.')
    result['accepted']=result['primary_learned_minus_control']['CI95'][0]>0 and all(v['mean']>0 and v['p_Holm4']<.05 for v in comparisons.values())
    n.put(out/'RESULTS.json',dict(status='COMPLETE',**result));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(expert_updates=250,selector_updates=250,control_updates=1000,wall_seconds=10800,evaluation_cases=1024,
       training='Three independently seeded complete-policy native-EIG continuations,then frozen experts/router-only exact categorical-Q expectation',
       specialization='Unproven until evaluated;no fabricated role-specific reward or artificial diversity loss',
       initial_policy='Every expert copies complete original V2;no shared final action head',
       primary='learned categorical V3 minus simulator-budget-matched original V2 continuation',
       case_seed=202611133201,routing_seed=202611133202,contrast_seed=202611133203,contrasts=16000000)
    import subprocess
    cfg['source_commit']=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip();n.put(out/'CONFIG.json',cfg)
    files=[*sorted(pathlib.Path(__file__).parent.glob('*.*')),*sorted((audit.BASE/'source').glob('*.py')),
           *sorted((n.P/'source').rglob('*.py')),audit.PARENT/'train_router.py',audit.BASE/'arms/candidate/actor.pt',audit.BASE/'arms/candidate/runtime.pkl']
    protected={str(f):n.sha(f) for f in files if f.is_file()};n.put(out/'PRE_RUN_MANIFEST.json',protected)
    n.original.setup()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds']
    try:
        for k in range(3):
            state=n.load_parent();a=state['algorithm'];n.original.restore(state['rng'])
            result=phase(a,f'expert{k+1}',cfg['expert_updates'],out,start,cfg['wall_seconds'],seed=202611133100+k)
            if result['status']!='COMPLETE':raise Stop(result['status'])
            del a,state;gc.collect();torch.cuda.empty_cache()
        state=n.load_parent();a=state['algorithm'];n.original.restore(state['rng']);before=migrate_selector(a,out)
        result=phase(a,'selector',cfg['selector_updates'],out,start,cfg['wall_seconds'],seed=202611133110)
        assert before==n.digest({k:v for k,v in a.actor_net.state_dict().items() if k.startswith('experts.')})
        if result['status']!='COMPLETE':raise Stop(result['status'])
        del a,state;gc.collect();torch.cuda.empty_cache()
        state=n.load_parent();a=state['algorithm'];n.original.restore(state['rng'])
        result=phase(a,'control',cfg['control_updates'],out,start,cfg['wall_seconds'])
        if result['status']!='COMPLETE':raise Stop(result['status'])
        del a,state;gc.collect();torch.cuda.empty_cache()
        result=evaluate(out,deadline,cfg['evaluation_cases'])
        assert all(n.sha(f)==v for f,v in protected.items())
        n.put(out/'STATUS.json',dict(status='COMPLETE',accepted=result['accepted'],historical_artifacts_preserved=True,seconds=time.monotonic()-start))
    except Stop as exc:n.put(out/'STATUS.json',dict(status='INCOMPLETE',reason=str(exc)))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()

if __name__=='__main__':main()
