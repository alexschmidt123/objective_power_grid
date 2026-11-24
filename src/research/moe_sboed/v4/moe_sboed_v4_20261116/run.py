"""One matched V4 seed101 pilot;unchanged native full-horizon EIG training."""
import argparse,copy,fcntl,hashlib,io,json,os,pathlib,subprocess,sys,time
M=pathlib.Path.home()/'Documents/objective_power_grid'
HERE=M/'src/research/moe_sboed_v4_20261116'
C=M/'configs/moe_sboed_v4_20261116.json'
O=M/'experiments/location_finding/moe_sboed_v4_20261116'
R=M/'reports/moe_sboed_v4_20261116'
sys.dont_write_bytecode=True
sys.path.insert(0,str(M/'src/research/hec_v3_study'))
sys.path.insert(0,str(HERE))
import engine
from runtime import n,setup,lease,sha,put,signal,load_module
from model import StudyMoE
from policy import V4Actor,ROLES
from hec_policy import COST
import numpy as np
import torch
from torch import nn
NAMES={'learned':'MoE-sBOED-V4','uniform':'MoE-sBOED-V4-uniform',
       'previous_learned':'Previous MoE-sBOED','previous_uniform':'Previous MoE-sBOED-uniform'}


def cfg():return json.loads(C.read_text())


def load_parent(device):
    assert sha(n.BASE/'arms/candidate/runtime.pkl')==n.RUNTIME_SHA==cfg()['parent_runtime_sha256']
    assert sha(n.BASE/'arms/candidate/actor.pt')==n.ACTOR_SHA==cfg()['parent_actor_sha256']
    old=torch.storage._load_from_bytes
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=device,weights_only=False)
    try:
        with (n.BASE/'arms/candidate/runtime.pkl').open('rb') as f:state=n.cloudpickle.load(f)
    finally:torch.storage._load_from_bytes=old
    a=state['algorithm']
    assert a.update==16001 and a.discount==1 and not a.use_PCE and not a.use_encoder
    assert a.n_newtraj==1000 and a.n_batch==10000 and len(a._runtime_buffer['params'])==1000000
    assert a.n_critic_update==5 and a.n_post_approx_update==5
    assert a.train.__func__ is type(a).train,'Do not inherit categorical actor-expectation patch'
    cp=torch.load(n.BASE/'arms/candidate/actor.pt',map_location=device,weights_only=True)['actor_state']
    assert all(torch.equal(v,cp[k]) for k,v in a.actor_net.state_dict().items())
    for op,_ in engine.opts(a):
        op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
    if hasattr(a,'_soft_tasks'):a._soft_tasks.enabled=False
    return state


def new_actor(parent,arm):
    # This temporary wrapper only exposes the frozen V2 query modules. It is never executed.
    template=StudyMoE(parent,202611150101,'uniform')
    engine.seed_all(cfg()['initialization_seed'] % (2**32))
    actor=V4Actor(template,arm)
    return actor


def build(arm,device):
    state=load_parent(device);a=state['algorithm'];actor=new_actor(a.actor_net,arm)
    a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor);a.actor_target_net.configure('frozen')
    engine.optimizer(a)
    if device=='cuda':n.original.restore(state['rng'])
    engine.seed_all(101)
    return state,a


def source_identity():
    return {str(p.relative_to(M)):sha(p) for p in [*HERE.glob('*.py'),C,M/'tools/moe_sboed_v4_pilot.py']}


def prepare():
    O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    path=O/'development_validation_bank.npz';assert not path.exists()
    rng=np.random.default_rng(cfg()['evaluation']['seed'])
    np.savez(path,truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)))
    put(O/'BANK_IDENTITY.json',dict(sha256=sha(path),evaluation=cfg()['evaluation'],purpose='Separate held-out V4 development-validation bank;original test bank untouched'))
    put(O/'SOURCE_MANIFEST.json',source_identity())
    print(json.dumps(dict(out=str(O),bank_sha256=sha(path))))


def bank():
    p=O/'development_validation_bank.npz';assert sha(p)==json.loads((O/'BANK_IDENTITY.json').read_text())['sha256']
    d=np.load(p);assert d['truth'].shape==(2000,4) and d['noise'].shape==(2000,30) and d['contrasts'].shape==(1000000,4)
    return d


def parameter_counts(model):
    result={}
    for name,module in [('query',model.query),*[(f'E{k+1}',e) for k,e in enumerate(model.experts)],
                        ('router',model.router),('action_head',model.action_head),('shared_small',model.shared_small)]:
        result[name]=dict(total=sum(p.numel() for p in module.parameters()),trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
                          buffer_elements=sum(b.numel() for b in module.buffers()))
    result['ALL']=dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       buffer_elements=sum(b.numel() for b in model.buffers()))
    return result


def gradients(model):
    result={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
            group='.'.join(name.split('.')[:2]);result[group]=result.get(group,0.)+float(p.grad.square().sum())
        else:assert p.grad is None,name
    return {k:v**.5 for k,v in result.items()}


def audit():
    torch.set_num_threads(2)
    state=load_parent('cpu');parent=state['algorithm'].actor_net
    learned=new_actor(parent,'learned');uniform=new_actor(parent,'uniform')
    assert n.digest(learned.state_dict())==n.digest(uniform.state_dict())
    forbidden={'SGActor','HECActor','V2Actor','StudyMoE','FeatureMoE','SharedDecoder','SingleFeaturePolicy','DecisionMoE'}
    assert not any(type(m).__name__ in forbidden for m in learned.modules())
    assert not hasattr(learned,'base') and not hasattr(learned,'decoder') and not hasattr(learned.query,'base')
    assert not any('first_projection' in k or 'response_projection' in k for k in learned.state_dict())
    assert [(m.in_features,m.out_features) for m in learned.action_head if isinstance(m,nn.Linear)]==[(128,32),(32,2)]
    assert [(m.in_features,m.out_features) for m in learned.shared_small if isinstance(m,nn.Linear)]==[(117,16),(16,2)]
    assert all(not e.normalization.elementwise_affine for e in learned.experts)
    rng=np.random.default_rng(20261116401);histories=[]
    for t in range(30):
        x=torch.zeros(2,117);x[:,t]=1
        x[:,30:30+2*t]=torch.tensor(rng.uniform(-4,4,(2,2*t)),dtype=torch.float32)
        x[:,88:88+t]=torch.tensor(rng.normal(size=(2,t)),dtype=torch.float32)
        histories.append(x)
    x=torch.cat(histories)
    learned.configure('joint');uniform.configure('joint')
    head_inputs=[]
    h=learned.action_head.register_forward_pre_hook(lambda module,args:head_inputs.append(args[0].detach().clone()))
    action,info=learned(x,collect=True);h.remove()
    assert action.shape==(60,2) and info['z'].shape==(60,4,128)
    assert torch.isfinite(info['z']).all() and torch.isfinite(action).all() and (action.abs()<=4).all()
    assert len(head_inputs)==1 and torch.equal(head_inputs[0],info['routed'].detach())
    assert info['correction'].abs().max()<=.20000001
    assert (action-info['a_moe']).abs().max()<=.200001
    info['gates'].retain_grad()
    critic=state['algorithm'].critic_net
    loss=-critic(torch.cat([x,action],-1)).mean();loss.backward()
    grad0=gradients(learned)
    assert torch.isfinite(info['gates'].grad).all() and info['gates'].grad.norm()>0
    assert sum(v*v for k,v in grad0.items() if k.startswith('router.'))>0
    for k in range(4):assert grad0[f'experts.{k}']>0
    # Zero-initialized router output bootstraps first-layer gradients after one actual critic-gradient step.
    op=torch.optim.Adam(learned.parameters(),lr=cfg()['actor_lr'])
    op.step();op.zero_grad()
    (-critic(torch.cat([x,learned(x)],-1)).mean()).backward()
    grad1=gradients(learned);assert grad1['router.0']>0 and grad1['router.2']>0
    learned=new_actor(parent,'learned').eval().requires_grad_(False)
    uniform=new_actor(parent,'uniform').eval().requires_grad_(False)
    with torch.no_grad():
        original,inf=learned(x,collect=True);other,ui=uniform(x,collect=True)
        torch.testing.assert_close(original,other,atol=0,rtol=0)
        assert torch.equal(ui['gates'],torch.full_like(ui['gates'],.25))
        uniform.router[-1].weight.fill_(100);uniform.router[-1].bias.copy_(torch.tensor([-100.,100.,10.,-10.]))
        unchanged,ui2=uniform(x,collect=True)
        torch.testing.assert_close(unchanged,other,atol=0,rtol=0)
        assert torch.equal(ui2['gates'],ui['gates'])
        # Perturb only weak-shared parameters;expert/router/MoE outputs must remain identical.
        learned.shared_small[-1].bias.fill_(100)
        changed,after=learned(x,collect=True)
        for key in ['z','raw','gates','a_moe']:torch.testing.assert_close(inf[key],after[key],atol=0,rtol=0)
        assert after['correction'].abs().max()<=.20000001 and (changed-after['a_moe']).abs().max()<=.200001
        zeros=torch.zeros_like(inf['z'])
        _,without=learned.decode(x,zeros,inf['gates'])
        torch.testing.assert_close(without['a_moe'],without['a_moe'][:1].expand_as(without['a_moe']),atol=0,rtol=0)
        learned.action_head[-1].weight.zero_();learned.action_head[-1].bias.fill_(100)
        clipped,ci=learned(x,collect=True)
        assert (clipped==4).all() and (ci['unclipped']>4).all()
        # No future-slot leakage.
        clean=new_actor(parent,'learned').eval()
        future=x.clone()
        for row,t in enumerate(x[:,:30].argmax(-1).tolist()):future[row,30+2*t:88]=123.;future[row,88+t:]=456.
        torch.testing.assert_close(clean(x),clean(future),atol=0,rtol=0)
        sensitivities=[]
        _,di=clean(x,collect=True)
        for k in range(4):
            w=.9*di['gates'];w[:,k]+=.1
            aa,_=clean.decode(x,di['z'],w)
            sensitivities.append(float((aa-original).norm(dim=-1).mean()))
        assert all(v>0 for v in sensitivities)
    # One full disposable native update per arm,using production configure/gradient assertions.
    a=state['algorithm'];native=[];tiny_buffer={k:v[:8].clone() for k,v in a._runtime_buffer.items()}
    class Logger:
        def __init__(self):self.logger=self;self.tabular=engine.Table()
        def push_prefix(self,*args):pass
        def pop_prefix(self,*args):pass
        def log(self,*args):pass
        def dump_all(self):pass
    logger=Logger();a.dowel=a.post_approx.dowel=logger;a.save_path=None
    a.n_newtraj=a.n_batch=4;a.n_buffer_init=8;a.n_buffer_max=16;a.n_post_approx_update=a.n_critic_update=1
    a.buffer_device='cpu';a._runtime_buffer=tiny_buffer;a._runtime_p_max=torch.arange(1,17,dtype=torch.float32);a._runtime_p=[]
    profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile']
    tiny=dict(profile,discount=1.,new_trajectories=4,replay_batch=4,initial_replay=8,replay_capacity=16,critic_updates=1,posterior_updates=1)
    for arm in ['learned','uniform']:
        actor=new_actor(parent,arm);actor.configure('warmup');a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor);a.actor_target_net.configure('frozen')
        engine.optimizer(a);before=n.digest(actor.state_dict());base=int(a.update)
        def check_gradients(*args):gradients(actor)
        a.actor_optimizer.register_step_pre_hook(check_gradients)
        n.original.train_native(a,logger,tiny,target=base+1)
        assert int(a.update)==base+1 and n.digest(actor.state_dict())!=before
        if arm=='uniform':assert all(p.grad is None for p in actor.router.parameters())
        native.append(dict(arm=arm,updates=1,reduced_disposable_fixture=True))
    counts={}
    for arm in ['learned','uniform']:
        actor=new_actor(parent,arm);counts[arm]={}
        for phase in ['warmup','joint']:
            actor.configure(phase);counts[arm][phase]=parameter_counts(actor)
    d=dict(status='PASS',device='CPU',histories=60,all30_stages=True,source_parent_stage_only=bool(getattr(parent,'stage_only',False)),
        no_old_physical_policy_module=True,no_shared_history_head_bypass=True,no_old_decoder_projection_buffers=True,
        weak_shared_isolated_and_bounded=True,uniform_invariant_to_router_perturbation=True,
        paired_initial_parameters_and_actions_bitwise_equal=True,no_future_observation_leakage=True,
        initial_gradient_norms=grad0,gradient_norms_after_disposable_step=grad1,
        initial_router_input_layer_zero_is_intentional=True,expert_weight_action_sensitivity=sensitivities,
        parameter_counts=counts,disposable_native_updates=native,original_test_bank_read=False)
    put(O/'AUDIT_CPU.json',d);print(json.dumps(d,indent=2))


@torch.no_grad()
def evaluate(actor,out,label,deadline):
    out.mkdir(parents=True,exist_ok=False)
    d=bank();lib=n.load_library();scorer_path=n.BASE/'source/frozen_evaluator.py'
    assert sha(scorer_path)==cfg()['evaluation']['scorer_sha256']
    scorer=load_module('v4_frozen_eig_scorer',scorer_path).contrast_scores
    contrasts=torch.tensor(d['contrasts'],device='cuda');scores=[];curves=[];details={}
    latency=scoring=0.;negative_calls=0;start=time.monotonic();actor.eval()
    for first in range(0,2000,16):
        engine.guard(deadline)
        th=torch.tensor(d['truth'][first:first+16],device='cuda');noise=torch.tensor(d['noise'][first:first+16],device='cuda')
        ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,device='cuda',dtype=torch.float64)
        local={}
        for t in range(30):
            x=lib.packed(ds,ys,t);torch.cuda.synchronize();tick=time.monotonic()
            if isinstance(actor,V4Actor):action,info=actor(x,collect=True)
            else:action=actor(x)
            torch.cuda.synchronize();latency+=time.monotonic()-tick
            assert torch.isfinite(action).all() and (action.abs()<=4.000001).all()
            if isinstance(actor,V4Actor):
                assert info['correction'].abs().max()<=.20000001
                assert (action-info['a_moe']).abs().max()<=.200001
                if actor.arm=='uniform':assert torch.equal(info['gates'],torch.full_like(info['gates'],.25))
                values=dict(gates=info['gates'],normalized_rms=info['z'].square().mean(-1).sqrt(),
                    raw_rms=info['raw'].square().mean(-1).sqrt(),a_moe=info['a_moe'],correction=info['correction'],
                    clipped=(info['unclipped'].abs()>4),shared_dominates=(info['correction'].abs()>info['a_moe'].abs()))
                effects=[]
                for k in range(4):
                    gates=.9*info['gates'];gates[:,k]+=.1
                    changed,_=actor.decode(x,info['z'],gates)
                    effects.append((changed-action).norm(dim=-1))
                values['weight_action_sensitivity']=torch.stack(effects,-1)
                for k,v in values.items():local.setdefault(k,[]).append(v.cpu().numpy())
            ds[:,t]=action;ys[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*noise[:,t]
        torch.cuda.synchronize();tick=time.monotonic();bench=type('Benchmark',(),{'sigma':.5,'calls':0})()
        score,_=scorer(bench,dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th),contrasts,16384,'cuda')
        torch.cuda.synchronize();scoring+=time.monotonic()-tick;negative_calls+=int(bench.calls)
        scores.extend(score[:,-1].numpy());curves.append(score.numpy())
        for k,v in local.items():details.setdefault(k,[]).append(np.stack(v,1))
        put(out/'STATUS.json',dict(status='EVALUATING',method=label,completed=first+len(th),total=2000))
    score=np.asarray(scores);assert len(score)==2000 and np.isfinite(score).all()
    np.save(out/'scores.npy',score);np.save(out/'curves.npy',np.concatenate(curves))
    result=dict(status='COMPLETE',method=label,seed=101,**engine.summaries(score),seconds=time.monotonic()-start,
        evaluation=cfg()['evaluation'],bank_sha256=sha(O/'development_validation_bank.npz'),
        actor_batch16_seconds=latency,actor_amortized_seconds_per_design=latency/60000,scorer_seconds=scoring,
        simulator_observations=60000,negative_likelihood_calls=negative_calls)
    if details:
        arrays={k:np.concatenate(v) for k,v in details.items()};np.savez_compressed(out/'diagnostics.npz',**arrays)
        g=arrays['gates'];cor=abs(arrays['correction']);moe=abs(arrays['a_moe'])
        result['diagnostics']=dict(mean_weights=g.mean((0,1)).tolist(),overall_weight_std=g.std((0,1),ddof=1).tolist(),
            stage_mean_weights=g.mean(0).tolist(),stage_history_weight_std=g.std(0,ddof=1).tolist(),
            normalized_representation_rms_mean=arrays['normalized_rms'].mean((0,1)).tolist(),
            normalized_representation_rms_min=arrays['normalized_rms'].min((0,1)).tolist(),
            normalized_representation_rms_max=arrays['normalized_rms'].max((0,1)).tolist(),
            weight_action_sensitivity_mean=arrays['weight_action_sensitivity'].mean((0,1)).tolist(),
            stage_weight_action_sensitivity_mean=arrays['weight_action_sensitivity'].mean(0).tolist(),
            correction_abs_mean=float(cor.mean()),correction_abs_max=float(cor.max()),
            correction_abs_mean_by_coordinate=cor.mean((0,1)).tolist(),
            a_moe_abs_mean=float(moe.mean()),a_moe_abs_max=float(moe.max()),
            a_moe_abs_mean_by_coordinate=moe.mean((0,1)).tolist(),
            clipping_coordinate_fraction=float(arrays['clipped'].mean()),
            shared_dominates_coordinate_fraction=float(arrays['shared_dominates'].mean()),
            shared_dominates_fraction_by_coordinate=arrays['shared_dominates'].mean((0,1)).tolist(),
            interpretation='Shared dominance means |correction|>|a_moe|,often when a_moe is near zero;correction still <=.2 and has no later gain.')
    put(out/'RESULTS.json',result);return result


def worker(arm,seconds):
    start=time.monotonic();out=O/arm;out.mkdir();setup()
    source=json.loads((O/'SOURCE_MANIFEST.json').read_text());assert source_identity()==source
    put(out/'CONFIG.json',dict(method=NAMES[arm],seed=101,arm=arm,protocol=cfg(),protocol_sha256=sha(C),source_manifest=source))
    with lease(),engine.Resources(out):
        try:
            state,a=build(arm,'cuda');actor=a.actor_net
            put(out/'INITIALIZATION.json',dict(actor_digest=n.digest(actor.state_dict()),critic_digest=n.digest(a.critic_net.state_dict()),
                posterior_digest=n.digest({f'{t}.{i}.{k}':v for t,nets in a.post_approx.poi_post_nets.items() for i,net in enumerate(nets) for k,v in net.state_dict().items()}),
                parent_actor_sha256=n.ACTOR_SHA,parent_runtime_sha256=n.RUNTIME_SHA,parent_update=16001,
                actor_lr=a.actor_optimizer.param_groups[0]['lr'],inherited_replay_episodes=len(a._runtime_buffer['params']),
                weak_coefficient=.2,initialization_seed=cfg()['initialization_seed']))
            counts={};phases=[];COST.clear();trainstart=time.monotonic()
            actor.train()
            for phase,steps in [('warmup',200),('joint',1800)]:
                actor.configure(phase);counts[phase]=parameter_counts(actor)
                phases.append(engine.train_phase(a,out,phase,steps,start,start+min(seconds-900,12600)))
            training_seconds=time.monotonic()-trainstart;model_calls=dict(COST)
            with (out/'runtime.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng(),method=NAMES[arm],seed=101),f,protocol=5)
            actor.configure('frozen');actor.eval()
            evaluation=evaluate(actor,out/'evaluation',NAMES[arm],start+min(seconds-60,14100))
            assert source_identity()==source
            assert sha(n.BASE/'arms/candidate/runtime.pkl')==n.RUNTIME_SHA and sha(n.BASE/'arms/candidate/actor.pt')==n.ACTOR_SHA
            logs=[json.loads(line) for line in (out/'TRAINING_METRICS.jsonl').read_text().splitlines()]
            groups={k:[r['gradient_group_norms'][k] for r in logs if k in r['gradient_group_norms']] for k in set().union(*(r['gradient_group_norms'] for r in logs))}
            gradstats={k:dict(mean=float(np.mean(v)),max=float(max(v)),min=float(min(v)),nonzero=sum(x>0 for x in v),snapshots=len(v)) for k,v in groups.items()}
            put(out/'RESULTS.json',dict(status='COMPLETE',method=NAMES[arm],seed=101,updates=2000,phases=phases,
                training_seconds=training_seconds,total_seconds=time.monotonic()-start,new_training_episodes=2000000,
                new_training_observations=60000000,parameter_counts=counts,training_gradient_norms=gradstats,
                training_model_derived_calls=model_calls,evaluation=evaluation,
                final_actor_sha256=sha(out/'joint_actor.pt'),historical_artifacts_preserved=True))
            put(out/'STATUS.json',dict(status='COMPLETE',method=NAMES[arm],seed=101,completed_updates=2000))
        except Exception as e:
            if 'actor' in locals():torch.save(dict(actor_state=actor.state_dict(),incomplete=True,error=repr(e)),out/'incomplete_actor.pt')
            put(out/'STATUS.json',dict(status='FAILED_OR_CAPPED',error=repr(e),seconds=time.monotonic()-start));raise


def previous(arm,seconds):
    start=time.monotonic();out=O/arm;out.mkdir();setup()
    with lease(),engine.Resources(out):
        state=load_parent('cuda');a=state['algorithm'];kind='learned' if arm=='previous_learned' else 'uniform'
        actor=StudyMoE(a.actor_net,202611150101,kind).cuda()
        checkpoint=M/f'experiments/location_finding/hec_v3_study_20261115/seed101_{kind}/joint_actor.pt'
        h=sha(checkpoint);actor.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['actor_state']);actor.configure('frozen');actor.eval()
        ev=evaluate(actor,out/'evaluation',NAMES[arm],start+seconds-30)
        assert sha(checkpoint)==h
        put(out/'RESULTS.json',dict(status='COMPLETE',method=NAMES[arm],evaluation=ev,checkpoint=str(checkpoint),checkpoint_sha256=h,training_updates=0))


def report():
    rows=[];lookup={}
    for arm in NAMES:
        path=O/arm/'RESULTS.json'
        if path.exists():
            d=json.loads(path.read_text());rows.append(dict(method=NAMES[arm],EIG=d['evaluation']['mean'],MCSE=d['evaluation']['MCSE'],score_path=str(O/arm/'evaluation/scores.npy')));lookup[arm]=rows[-1]
    contrasts={};expand=None
    if 'learned' in lookup:
        a=np.load(lookup['learned']['score_path'])
        for arm,row in lookup.items():
            if arm!='learned':contrasts[NAMES[arm]]=engine.summaries(a-np.load(row['score_path']))
    if len(rows)==4:
        aa,bb=[json.loads((O/arm/'INITIALIZATION.json').read_text()) for arm in ['learned','uniform']]
        assert aa['actor_digest']==bb['actor_digest'] and aa['critic_digest']==bb['critic_digest'] and aa['posterior_digest']==bb['posterior_digest']
        expand=contrasts[NAMES['uniform']]['CI95'][0]>0 and contrasts[NAMES['previous_learned']]['CI95'][0]>-cfg()['competitive_margin_nats']
    result=dict(status='COMPLETE' if len(rows)==4 else 'PENDING',rows=rows,paired_V4_differences=contrasts,
        recommend_five_seeds=expand,automatic_expansion=False,bank_identity=json.loads((O/'BANK_IDENTITY.json').read_text()),
        limitation='One paired continuation seed on development-validation bank;case MC uncertainty is not across-seed robustness. No exact-MI or from-scratch compute superiority claim.')
    R.mkdir(exist_ok=True,parents=True);put(R/'RESULTS.json',result)
    text='# V4 seed101 pilot results\n\n| Method | EIG (nats) | MCSE |\n|---|---:|---:|\n'
    for row in rows:text+=f"| {row['method']} | {row['EIG']:.5f} | {row['MCSE']:.5f} |\n"
    text+='\nPaired V4 differences:\n\n```json\n'+json.dumps(contrasts,indent=2)+'\n```\n\n'
    text+='Five-seed recommendation:'+('PENDING' if expand is None else 'YES,consider validation;not automatically scheduled' if expand else 'NO,predeclared pilot criteria not met')+'\n\n'+result['limitation']+'\n'
    (R/'RESULTS.md').write_text(text)
    # Full stage-wise routing/shared diagnostics remain linked to their raw cases.
    diagnostics={}
    for arm in ['learned','uniform']:
        path=O/arm/'RESULTS.json'
        if path.exists():
            d=json.loads(path.read_text());diagnostics[NAMES[arm]]=dict(evaluation=d['evaluation'].get('diagnostics'),training_gradient_norms=d['training_gradient_norms'],parameter_counts=d['parameter_counts'],training_seconds=d['training_seconds'])
    put(R/'DIAGNOSTICS.json',diagnostics)


def controller():
    O.mkdir(exist_ok=True,parents=True);lock=(O/'.controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((O/'AUDIT_CPU.json').read_text())['status']=='PASS'
    assert not (O/'LAUNCH.json').exists(),'Do not restart/duplicate an existing attempt'
    put(O/'LAUNCH.json',dict(pid=os.getpid(),started=time.time(),protocol_sha256=sha(C)))
    try:
        for arm in ['learned','uniform','previous_learned','previous_uniform']:
            assert not (O/arm).exists(),'Preserve existing component;no automatic retry'
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB',next=NAMES[arm]));time.sleep(20)
            seconds=14400 if arm in ['learned','uniform'] else 1800
            with (O/(arm+'.log')).open('x') as f:
                child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(seconds-30),sys.executable,'-u',str(M/'tools/moe_sboed_v4_pilot.py'),'--task','train' if arm in ['learned','uniform'] else 'previous','--arm',arm,'--seconds',str(seconds)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),worker_pid=child.pid,method=NAMES[arm],arm=arm,seed=101,started=time.time()))
                put(O/'STATUS.json',dict(status='RUNNING',method=NAMES[arm],seed=101,pending=list(NAMES)[list(NAMES).index(arm)+1:]))
                code=child.wait()
            if code:raise RuntimeError(f'{arm} failed/capped:returncode{code};no automatic retries or expansion')
            report()
        put(O/'STATUS.json',dict(status='COMPLETE',report=str(R/'RESULTS.md'),automatic_expansion=False))
    except Exception as e:
        put(O/'STATUS.json',dict(status='FAILED_OR_CAPPED',error=repr(e),automatic_expansion=False));report();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['prepare','audit','controller','train','previous','report'],required=True)
    p.add_argument('--arm',choices=list(NAMES),default='learned');p.add_argument('--seconds',type=float,default=14400)
    a=p.parse_args()
    if a.task=='prepare':prepare()
    elif a.task=='audit':audit()
    elif a.task=='controller':controller()
    elif a.task=='train':worker(a.arm,a.seconds)
    elif a.task=='previous':previous(a.arm,a.seconds)
    elif a.task=='report':report()


if __name__=='__main__':main()
