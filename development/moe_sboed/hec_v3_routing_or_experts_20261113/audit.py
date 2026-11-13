"""Frozen V3 audit: routing quality versus expert design, no training."""
import argparse,gc,hashlib,importlib.util,io,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
V3=WORK/'development/moe_sboed/hec_v3_decision_20261113';sys.path.insert(0,str(V3))
from common import n
from policy import DecisionMoE
from resolution import signal,nested_scores,paired
OLD=WORK/'development/moe_sboed/hec_v2_routing_gap_20261113/diagnostic.py'
spec=importlib.util.spec_from_file_location('posterior_reference',OLD);posterior_reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(posterior_reference)
PILOT=ROOT/'scratch_runs/hec_v3_decision_20261113_pilot'

def load():
    receipt=json.loads((PILOT/'selector/RESULT.json').read_text())
    runtime=PILOT/'selector/runtime.pkl';actor_file=PILOT/'selector/actor.pt'
    assert n.sha(runtime)==receipt['runtime_sha256'] and n.sha(actor_file)==receipt['actor_sha256']
    old=torch.storage._load_from_bytes
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=lambda s,l:s.cuda(0) if l.startswith('cuda') else s,weights_only=False)
    try:
        with runtime.open('rb') as f:state=n.cloudpickle.load(f)
    finally:torch.storage._load_from_bytes=old
    a=state['algorithm'];assert isinstance(a.actor_net,DecisionMoE)
    a.actor_net.eval().requires_grad_(False);a.critic_net.eval().requires_grad_(False)
    saved=torch.load(actor_file,map_location='cuda',weights_only=True)['actor_state']
    assert all(torch.equal(v,a.actor_net.state_dict()[k]) for k,v in saved.items())
    return a

@torch.no_grad()
def branch(actor,lib,ds,ys,stage,theta,noise,uniforms,deadline):
    m=len(theta);k=3
    actions=ds.expand(m,k,30,2).clone().reshape(-1,30,2);obs=ys.expand(m,k,30).clone().reshape(-1,30)
    actions[:,stage:]=0;obs[:,stage:]=0
    th=theta[:,None].expand(m,k,4).reshape(-1,4)
    eps=noise[:,None].expand(m,k,30).reshape(-1,30)
    u=uniforms[:,None].expand(m,k,30).reshape(-1,30)
    choices=torch.arange(3,device=theta.device).repeat(m)
    for t in range(stage,30):
        if time.monotonic()>deadline:raise TimeoutError('Locked audit cap')
        x=lib.packed(actions,obs,t)
        if t==stage:
            _,proposals,_,_=actor.proposals_and_weights(x)
            action=proposals[torch.arange(len(x),device=x.device),choices]
        else:action=actor(x,uniforms=u[:,t])
        assert torch.isfinite(action).all() and bool((action.abs()<=4.000001).all())
        actions[:,t]=action;obs[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*eps[:,t]
    assert torch.equal(actions[:,:stage],ds[:,:stage].expand(m*k,-1,-1))
    assert torch.equal(obs[:,:stage],ys[:,:stage].expand(m*k,-1))
    return actions,obs,th

def comparisons(selection,test,weights,critic):
    choose=selection.mean(1).argmax(-1);qpick=critic.argmax(-1);routerpick=weights.argmax(-1)
    fixed=int(selection.mean((0,1)).argmax());means=test.mean(1);row=np.arange(len(test))
    sampled_value=(means*weights).sum(-1);uniform_value=means.mean(-1)
    return dict(selection_chosen_experts=(choose+1).tolist(),critic_chosen_experts=(qpick+1).tolist(),
       router_argmax_experts=(routerpick+1).tolist(),selection_chosen_fixed=fixed+1,
       primary_selection_choice_minus_learned=paired(means[row,choose],sampled_value),
       expert_design_conditional_minus_fixed=paired(means[row,choose],means[:,fixed]),
       routing_critic_choice_minus_learned=paired(means[row,qpick],sampled_value),
       routing_selection_choice_minus_critic_choice=paired(means[row,choose],means[row,qpick]),
       learned_minus_uniform=paired(sampled_value,uniform_value),
       critic_agreement_with_selection=float((choose==qpick).mean()),
       router_agreement_with_selection=float((choose==routerpick).mean()),
       test_mean_by_expert=means.mean(0).tolist(),
       interpretation='Expert design: independently validated conditional advantages. Routing quality: exploiting those advantages;critic ranking/selector optimization are subissues. Agreement alone is not a utility test.')

def tests(actor,lib):
    x=torch.zeros(1,117,device='cuda');x[:,14]=1
    theta=torch.tensor([[.2,-.4,.8,1.]],device='cuda',dtype=torch.float64)
    zero=torch.zeros(1,30,device='cuda',dtype=torch.float64);u=torch.full((1,30),.5,device='cuda')
    with torch.no_grad():
        ds=torch.zeros(1,30,2,device='cuda');ys=zero.clone()
        actions,obs,th=branch(actor,lib,ds,ys,14,theta,zero,u,float('inf'))
        assert torch.equal(actions[:,:14],ds[:,:14].expand(3,-1,-1))
        torch.testing.assert_close(obs[:,14:],signal(th.reshape(3,1,2,2),actions[:,14:].double()),atol=1e-12,rtol=1e-12)
    # Holdout rankings must not select either conditional or fixed alternatives.
    selection=np.zeros((4,2,3));selection[:,:,1]=4
    test=np.zeros((4,3,3));test[:,:,2]=9
    result=comparisons(selection,test,np.full((4,3),1/3),np.zeros((4,3)))
    assert result['selection_chosen_experts']==[2]*4 and result['selection_chosen_fixed']==2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(smoke=args.smoke,histories=3 if args.smoke else 24,stages=[3,14,26],selection_trials=2 if args.smoke else 32,
       test_trials=3 if args.smoke else 64,selection_contrasts=31 if args.smoke else 4000000,
       test_contrasts=127 if args.smoke else 16000000,particles=[8192] if args.smoke else [2097152,8388608],
       minimum_ESS=1 if args.smoke else 256,max_weight=1 if args.smoke else .05,
       seed=202611133701,wall_seconds=300 if args.smoke else 7200,
       conditioning='First24 saved learned-V3 histories,8 per prefix stage;generating truth never read',
       primary='Heldout sPCE value of selection-bank preferred first expert minus learned categorical expectation',
       native_critic_check='Also score unchanged native terminal logposterior minus prefix logposterior;separate critic-target misranking from evaluation-proxy mismatch',
       categories=['routing_quality','expert_design'])
    n.put(out/'CONFIG.json',cfg)
    files=[*sorted(pathlib.Path(__file__).parent.glob('*.py')),*sorted(V3.glob('*.*')),OLD,
       PILOT/'selector/runtime.pkl',PILOT/'selector/actor.pt',PILOT/'selector/RESULT.json',PILOT/'LEARNED_HISTORIES.npz']
    protected={str(f):n.sha(f) for f in files if f.is_file()};n.put(out/'PRE_RUN_MANIFEST.json',protected)
    n.original.setup()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds']
    try:
        a=load();actor=a.actor_net;lib=n.load_library();tests(actor,lib)
        n.put(out/'VALIDATION.json',dict(status='PASS',prefix_chronology=True,independent_selection=True,native_critic_loaded=True,training_steps=0))
        saved=np.load(PILOT/'LEARNED_HISTORIES.npz');h=cfg['histories']
        ds=torch.tensor(saved['actions'][:h],device='cuda');ys=torch.tensor(saved['observations'][:h],device='cuda')
        stages=np.repeat(cfg['stages'],h//3);weights=[];critic=[];draws=[];quality=[]
        for i,t in enumerate(stages):
            x=lib.packed(ds[i:i+1],ys[i:i+1],int(t))
            with torch.no_grad():
                clean,proposals,_,w=actor.proposals_and_weights(x)
                q=a.critic_net(torch.cat([clean.expand(3,-1),proposals[0]],-1)).flatten()
            weights.append(w[0].cpu().numpy());critic.append(q.cpu().numpy());entry={};row={}
            for split_index,split in enumerate(['selection','test']):
                for attempt,particles in enumerate(cfg['particles']):
                    seed=cfg['seed']+10000+100*i+10*split_index+attempt
                    theta,details=posterior_reference.posterior(actor.experts[0].core,x,ds[i:i+1],ys[i:i+1],int(t),cfg[split+'_trials'],particles,seed)
                    good=details['ESS']>=cfg['minimum_ESS'] and details['max_weight']<=cfg['max_weight']
                    if good:break
                entry[split]=theta;row[split]=dict(accepted=good,**details)
            draws.append(entry);quality.append(dict(history=i,stage=int(t),**row));n.put(out/'POSTERIOR_QUALITY.json',quality)
        if not all(q[s]['accepted'] for q in quality for s in ['selection','test']):
            n.put(out/'STATUS.json',dict(status='INCONCLUSIVE_POSTERIOR_QUALITY',training_steps=0));return
        weights=np.array(weights);critic=np.array(critic);np.savez_compressed(out/'ROUTER_CRITIC.npz',weights=weights,critic=critic,stages=stages)
        spce={};native={}
        for split_index,split in enumerate(['selection','test']):
            bank=np.random.default_rng(cfg['seed']+20000+split_index).standard_normal((cfg[split+'_contrasts'],4))
            n.put(out/f'{split}_BANK.json',dict(seed=cfg['seed']+20000+split_index,sha256=hashlib.sha256(bank.view(np.uint8)).hexdigest()))
            negatives=torch.tensor(bank,device='cuda');del bank;spce[split]=[];native[split]=[]
            for i,t in enumerate(stages):
                theta=draws[i][split];m=len(theta);rng=np.random.default_rng(cfg['seed']+30000+100*i+split_index)
                eps=torch.tensor(rng.standard_normal((m,30)),device='cuda')
                u=torch.tensor(rng.random((m,30)),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
                with torch.no_grad(),torch.device('cuda'):
                    act,obs,th=branch(actor,lib,ds[i:i+1],ys[i:i+1],int(t),theta,eps,u,deadline)
                    terminal=a.post_approx.log_poi_post(29,act.float(),obs.float()[...,None],th.float())
                    prefix=a.post_approx.log_poi_post(int(t)-1,act[:,:int(t)].float(),obs[:,:int(t)].float()[...,None],th.float())
                    native_value=(terminal-prefix).reshape(m,3).cpu().numpy()
                values=[]
                for first in range(0,len(act),16):values.append(nested_scores(act[first:first+16],obs[first:first+16],th[first:first+16],negatives,[cfg[split+'_contrasts']],deadline=deadline)[cfg[split+'_contrasts']].cpu().numpy())
                score=np.concatenate(values).reshape(m,3);spce[split].append(score);native[split].append(native_value)
                np.savez_compressed(out/f'{split}_history{i:02d}.npz',spce=score,native=native_value)
                n.put(out/'STATUS.json',dict(status='RUNNING',split=split,history=i+1,total=h,seconds=time.monotonic()-start));print(json.dumps(json.loads((out/'STATUS.json').read_text())),flush=True)
            del negatives
        result=dict(status='COMPLETE',smoke=args.smoke,
            spce=comparisons(np.array(spce['selection']),np.array(spce['test']),weights,critic),
            native_return=comparisons(np.array(native['selection']),np.array(native['test']),weights,critic),
            seconds=time.monotonic()-start,training_steps=0,
            categories=['routing_quality','expert_design'],
            limitations='24 diagnostic histories,finite importance posterior and contrast bounds,one checkpoint. Single initial decision only;no deployable oracle claim. Independent banks validate selection;rank disagreement alone is not a bug. Secondary comparisons exploratory/unadjusted. Existing V3 expert settings differ only by training seeds,not designed sBOED strategies.')
        assert all(n.sha(f)==v for f,v in protected.items())
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',historical_artifacts_preserved=True,training_steps=0,seconds=time.monotonic()-start))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()

if __name__=='__main__':main()
