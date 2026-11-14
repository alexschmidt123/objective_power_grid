"""Frozen V2-bank ablations to localize routing/expert-design loss."""
import argparse,copy,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
SOURCE=WORK/'development/moe_sboed/hec_v3_v2_pathways_20261113';sys.path.insert(0,str(SOURCE))
from common import n,audit
from pathway_policy import PathwayMoE
from resolution import nested_scores,paired,signal
PILOT=ROOT/'scratch_runs/hec_v3_v2_pathways_20261113_pilot'
PREVIOUS=ROOT/'scratch_runs/hec_v3_same_V2_bank_20261113_02'

def load_runtime():
    import io
    path=PILOT/'selector/runtime.pkl';receipt=json.loads((PILOT/'selector/RESULT.json').read_text());assert n.sha(path)==receipt['runtime_sha256']
    old=torch.storage._load_from_bytes
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=lambda s,l:s.cuda(0) if l.startswith('cuda') else s,weights_only=False)
    try:
        with path.open('rb') as f:state=n.cloudpickle.load(f)
    finally:torch.storage._load_from_bytes=old
    return state['algorithm']

@torch.no_grad()
def episode(model,parent,critic,lib,theta,noise,u,mode,k=None):
    b=len(theta);ds=torch.zeros(b,30,2,device='cuda');ys=torch.zeros(b,30,device='cuda',dtype=torch.float64)
    active=None;last=None;switches=0;counts=torch.zeros(4,device='cuda',dtype=torch.long)
    for t in range(30):
        x=lib.packed(ds,ys,t)
        if mode=='standalone':
            action=model(x);chosen=torch.full((b,),k,device='cuda',dtype=torch.long)
        else:
            proposals,g,legacy,response=model.components(x)
            if mode=='critic_greedy':
                q=torch.stack([critic(torch.cat([x,proposals[:,j]],-1)).squeeze(-1) for j in range(4)],-1)
                chosen=q.argmax(-1)
            elif mode=='argmax':chosen=g.argmax(-1)
            elif mode=='bridge':
                c=model.context(x)
                raw=parent.coordinator(torch.cat([c['s'],legacy.flatten(1)],-1))+parent.response_gate(response.flatten(1))
                weights=(10*(raw/10).tanh()).softmax(-1)
                packets=[expert.packet(c) for expert in model.experts];first=torch.stack([p[0] for p in packets],1)
                pre=parent.fusion[0](torch.cat([c['s'],(weights[...,None]*first).flatten(1)],-1))+parent.response_fusion((weights[...,None]*response).flatten(1))
                delta=parent.fusion[2](parent.fusion[1](pre)).tanh();action=c['a0']+4*delta-c['a0']*delta.abs()
                ds[:,t]=action;ys[:,t]=signal(theta.reshape(b,2,2),action.double())+.5*noise[:,t]
                continue
            else:
                if mode=='block3':refresh=t%3==0
                elif mode=='episode_commit':refresh=t==0
                elif mode in ['learned','initial']:refresh=True
                else:raise ValueError(mode)
                if refresh:active=(u[:,t,None]>=g.cumsum(-1)).sum(-1).clamp_max(3)
                chosen=active
            action=proposals[torch.arange(b,device='cuda'),chosen]
        assert torch.isfinite(action).all() and bool((action.abs()<=4.000001).all())
        counts+=torch.bincount(chosen,minlength=4)
        if last is not None:switches+=int((last!=chosen).sum())
        last=chosen;ds[:,t]=action;ys[:,t]=signal(theta.reshape(b,2,2),action.double())+.5*noise[:,t]
    return ds,ys,counts,switches

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(training_steps=0,cases=2000,contrasts=1000000,bank='exact original V2 bank,including saved routing uniforms',wall_seconds=1800,
       variants=['initial_router','argmax','critic_greedy','block3','episode_commit','V2_head_bridge',*['prepared_E'+str(k) for k in range(1,5)]],
       primary_tests='Frozen mechanistic diagnostics,not final independent method acceptance;Holm across listed primary ablation comparisons')
    n.put(out/'CONFIG.json',cfg)
    files=[*sorted(pathlib.Path(__file__).parent.glob('*.py')),*sorted(SOURCE.glob('*.py')),
       audit.BASE/'evaluation/development_validation_bank.npz',audit.BASE/'evaluation/routing_uniforms.npy',
       PILOT/'selector/actor.pt',PILOT/'selector/runtime.pkl',PILOT/'INITIAL_DISTILLED.pt',
       *[PILOT/f'expert{k}/actor.pt' for k in range(1,5)],PREVIOUS/'RESULTS.json']
    protected={str(f):n.sha(f) for f in files};n.put(out/'PRE_RUN_MANIFEST.json',protected)
    n.original.setup()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+1800
    try:
        a=load_runtime();model=a.actor_net.eval().requires_grad_(False);critic=a.critic_net.eval().requires_grad_(False)
        state=n.load_parent();parent=state['algorithm'].actor_net.eval().requires_grad_(False)
        prepared=PathwayMoE(parent).cuda();prepared.load_state_dict(torch.load(PILOT/'INITIAL_DISTILLED.pt',map_location='cuda',weights_only=True)['actor_state']);prepared.requires_grad_(False)
        initial=copy.deepcopy(prepared)
        for k in range(4):initial.experts[k].load_state_dict(model.experts[k].state_dict())
        expected=json.loads((PILOT/'selector/STARTED.json').read_text())['initial_actor_digest'];assert n.digest(initial.state_dict())==expected
        # Before expert-core training,the bridge must reproduce V2 exactly in shape/semantics.
        lib=n.load_library();data=np.load(audit.BASE/'evaluation/diagnostic_candidate_histories.npz')
        x=torch.tensor(data['X'][:2,[0,14,29]].copy(),device='cuda').reshape(-1,117)
        with torch.no_grad():
            c=prepared.context(x);_,g,_,response=prepared.components(x)
            packets=[e.packet(c) for e in prepared.experts];first=torch.stack([p[0] for p in packets],1)
            pre=parent.fusion[0](torch.cat([c['s'],(g[...,None]*first).flatten(1)],-1))+parent.response_fusion((g[...,None]*response).flatten(1))
            d=parent.fusion[2](parent.fusion[1](pre)).tanh();bridge=c['a0']+4*d-c['a0']*d.abs()
            torch.testing.assert_close(bridge,parent(x),atol=2e-6,rtol=2e-5)
        n.put(out/'VALIDATION.json',dict(status='PASS',initial_selector_digest_exact=True,prepared_bridge_matches_V2=True,stages=[0,14,29],training_steps=0))
        bank=np.load(audit.BASE/'evaluation/development_validation_bank.npz')
        theta=torch.tensor(bank['truth'],device='cuda');noise=torch.tensor(bank['noise'],device='cuda');negative=torch.tensor(bank['contrasts'],device='cuda')
        u=torch.tensor(np.load(audit.BASE/'evaluation/routing_uniforms.npy'),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
        results={};stats={}
        for name in cfg['variants']:
            if name=='initial_router':current=initial;mode='initial';single=None
            elif name.startswith('prepared_E'):current=prepared.standalone(int(name[-1])-1).eval().requires_grad_(False);mode='standalone';single=int(name[-1])-1
            else:current=model;mode='bridge' if name=='V2_head_bridge' else name;single=None
            values=[];counts=torch.zeros(4,device='cuda',dtype=torch.long);switches=0
            for first in range(0,2000,16):
                if time.monotonic()>deadline:raise TimeoutError('Ablation cap')
                ds,ys,c,s=episode(current,parent,critic,lib,theta[first:first+16],noise[first:first+16],u[first:first+16],mode,single)
                values.append(nested_scores(ds,ys,theta[first:first+16],negative,[1000000],deadline=deadline)[1000000].cpu().numpy());counts+=c;switches+=s
                n.put(out/'STATUS.json',dict(status='EVALUATING',variant=name,completed=min(first+16,2000),total=2000,seconds=time.monotonic()-start))
            results[name]=np.concatenate(values);np.save(out/f'{name}_scores.npy',results[name]);stats[name]=dict(mean=float(results[name].mean()),counts=counts.cpu().tolist(),switches=switches)
            n.put(out/'PARTIAL.json',stats);print('EVALUATED',name,json.dumps(stats[name]),flush=True)
        learned=np.load(PREVIOUS/'V3_scores.npy');v2=np.load(PREVIOUS/'V2_scores.npy')
        comparisons={name:paired(score,learned) for name,score in results.items() if not name.startswith('prepared_E')}
        for k in range(1,5):comparisons[f'native_training_E{k}']=paired(np.load(PREVIOUS/f'expert{k}_scores.npy'),results[f'prepared_E{k}'])
        previous=0
        for i,key in enumerate(sorted(comparisons,key=lambda key:comparisons[key]['p_two_sided'])):
            previous=max(previous,min(1,(len(comparisons)-i)*comparisons[key]['p_two_sided']));comparisons[key]['p_Holm']=previous
        result=dict(status='COMPLETE',statistics=stats,comparisons=comparisons,
           prepared_vs_V2={f'E{k}':paired(results[f'prepared_E{k}'],v2) for k in range(1,5)},
           seconds=time.monotonic()-start,training_steps=0,
           limitations='Frozen diagnostic reuse of original historical bank,finite1M prior-sPCE ceiling,one lineage. Critic-greedy mode is offline diagnostic,not production deployment. Bridge is an untrained assembly,not a new accepted model.')
        assert all(n.sha(f)==h for f,h in protected.items())
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',historical_artifacts_preserved=True,training_steps=0,seconds=time.monotonic()-start))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
