"""One paired seed101 categorical MoE pilot; never edits historical artifacts."""
import argparse, copy, difflib, fcntl, hashlib, inspect, io, json, os, pathlib
import subprocess, sys, textwrap, time, types

M = pathlib.Path.home()/'Documents/objective_power_grid'
OLD = M/'src/research/hec_v3_study'
HERE = M/'src/research/moe_sboed_categorical_pilot_20261116'
C = M/'configs/moe_sboed_categorical_pilot_20261116.json'
O = M/'experiments/location_finding/moe_sboed_categorical_pilot_20261116'
R = M/'reports/moe_sboed_categorical_pilot_20261116'
P = M/'experiments/location_finding/hec_v3_study_20261115/seed101_uniform'
sys.dont_write_bytecode = True
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE))
import engine
from runtime import n, setup, lease, sha, put, signal, load_module
from policy import DecisionMoE, SharedInference, PrivateExpert, StandalonePolicy, ROLES
from hec_policy import COST
import numpy as np
import torch
from torch import nn
NAMES = {'learned':'MoE-sBOED', 'uniform':'MoE-sBOED-uniform'}


def cfg(): return json.loads(C.read_text())


def seed(value):
    engine.seed_all(value % (2**32))


def load_parent(device):
    c = cfg()
    assert sha(P/'runtime.pkl') == c['parent_runtime_sha256']
    assert sha(P/'joint_actor.pt') == c['parent_actor_sha256']
    old = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
    try:
        with (P/'runtime.pkl').open('rb') as f: state = n.cloudpickle.load(f)
    finally: torch.storage._load_from_bytes = old
    a = state['algorithm']
    assert a.update == 18001 and a.discount == 1 and not a.use_PCE
    for op, _ in engine.opts(a):
        op._optimizer_step_pre_hooks.clear()
        op._optimizer_step_post_hooks.clear()
    return state


def initial_models(parent):
    seed(cfg()['initialization_seed'])
    inference = SharedInference(parent)
    experts = [PrivateExpert(parent, k) for k in range(4)]
    # No object or trainable parameter sharing across expert decision modules.
    parameter_sets = [{id(p) for p in e.parameters()} for e in experts]
    assert all(not (parameter_sets[i] & parameter_sets[j]) for i in range(4) for j in range(i+1,4))
    assert not any('base' in name or 'decoder' in name or 'fusion' in name for e in experts for name, _ in e.named_modules())
    return inference, experts


def configure_native(a, actor, categorical, out):
    a.actor_net = actor
    actor.configure()
    a.actor_target_net = copy.deepcopy(actor).requires_grad_(False)
    a.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg()['actor_lr'])
    a.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer, gamma=.9999)
    # Fresh on-policy initial replay; do not silently train these decisions on old collaborative replay.
    for name in ['_runtime_buffer', '_runtime_p_max', '_runtime_p']:
        if hasattr(a, name): delattr(a, name)
    a.n_newtraj, a.n_batch = 1000, 10000
    a.n_buffer_init, a.n_buffer_max = 10000, 1000000
    a.buffer_device = next(actor.parameters()).device
    profile = json.loads((n.P/'frozen_protocol.json').read_text())['profile']
    a.n_critic_update = profile['critic_updates']
    a.n_post_approx_update = profile['posterior_updates']
    # Five updates of each of30 stage posteriors =150 optimizer steps per actor update.
    assert a.n_critic_update == 5 and a.n_post_approx_update == 5
    a.use_PCE, a.discount = False, 1.
    a.save_path = None
    if categorical:
        patch_native_expectation(a, out)


def patch_native_expectation(a, out):
    """Change ONLY the native scalar actor loss to its categorical expectation.

    Rollout/target actions remain categorical samples. All native variational
    rewards, posterior/critic updates, replay and chronology remain untouched.
    """
    original = textwrap.dedent(inspect.getsource(type(a).train))
    old = 'designs = self.actor_net(X_actor)\n            X = torch.cat([X_actor, designs], dim=-1)\n            output = -self.critic_net(X).mean()'
    assert original.count(old) == 1, 'Native actor branch changed;refuse approximate patch'
    modified = original.replace(old, 'output = self.actor_net.expected_actor_loss(X_actor, self.critic_net)')
    modified = modified.replace('def train(', 'def categorical_native_train(', 1)
    namespace = {}
    exec(compile(modified, '<native_categorical_actor_expectation>', 'exec'), a.train.__func__.__globals__, namespace)
    a.train = types.MethodType(namespace['categorical_native_train'], a)
    if out is not None:
        (out/'NATIVE_ACTOR_EXPECTATION.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True), fromfile='unchanged_native_train', tofile='categorical_actor_expectation')))
        put(out/'NATIVE_PATCH.json', dict(original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            executed_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),
            change='Only actor loss:-E_k Q(h,a_k);categorical rollouts and stochastic target backup retained',
            objectives_unchanged=True))


class Logger:
    def __init__(self, a, out, steps, start, deadline):
        self.a, self.out, self.steps = a, out, steps
        self.base, self.start, self.deadline = int(a.update), start, deadline
        self.logger, self.tabular = self, engine.Table()
        self.counts, self.gradients = dict(actor=0, critic=0, posterior=0), {}
    def push_prefix(self,*args): pass
    def pop_prefix(self,*args): pass
    def log(self,*args): pass
    def dump_all(self):
        done = int(self.a.update)-self.base
        assert self.counts == dict(actor=done, critic=5*done, posterior=150*done)
        if done == 1 or done % 25 == 0 or done == self.steps:
            row = dict(status='TRAINING', completed_updates=done, target_updates=self.steps,
                seconds=time.monotonic()-self.start, counts=self.counts.copy(), gradient_group_norms=self.gradients,
                actor_lr=self.a.actor_optimizer.param_groups[0]['lr'])
            with (self.out/'TRAINING_METRICS.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            put(self.out/'STATUS.json', row)
            print(json.dumps(row), flush=True)
        engine.guard(self.deadline)


def train(a, out, steps, start, deadline):
    a.actor_net.configure()
    logger = Logger(a, out, steps, start, deadline)
    a.dowel = a.post_approx.dowel = logger
    for op, key in engine.opts(a):
        op._optimizer_step_pre_hooks.clear()
        op._optimizer_step_post_hooks.clear()
        op.register_step_post_hook(lambda *args, key=key: logger.counts.__setitem__(key, logger.counts[key]+1))
    def check(*args):
        values = {}
        for name, p in a.actor_net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                group = '.'.join(name.split('.')[:2])
                values[group] = values.get(group, 0.)+float(p.grad.square().sum())
            else: assert p.grad is None, name
        logger.gradients = {k:v**.5 for k,v in values.items()}
    a.actor_optimizer.register_step_pre_hook(check)
    frozen = engine.frozen_digest(a.actor_net)
    profile = json.loads((n.P/'frozen_protocol.json').read_text())['profile']
    profile.update(discount=1., initial_replay=10000)
    tick = time.monotonic()
    n.original.train_native(a, logger, profile, target=int(a.update)+steps)
    assert engine.frozen_digest(a.actor_net) == frozen
    for op,_ in engine.opts(a):
        op._optimizer_step_pre_hooks.clear()
        op._optimizer_step_post_hooks.clear()
    receipt = dict(status='COMPLETE', updates=steps, counts=logger.counts, seconds=time.monotonic()-tick,
        trainable_parameters=sum(p.numel() for p in a.actor_net.parameters() if p.requires_grad),
        initial_replay_episodes=10000, new_update_episodes=1000*steps,
        total_simulator_episodes=10000+1000*steps, simulator_observations=30*(10000+1000*steps),
        frozen_state_preserved=True, objective='Native full-horizon variational EIG,use_PCE=False,discount1')
    put(out/'TRAINING.json', receipt)
    return receipt


def load_bank():
    c = cfg()['evaluation']
    path = O/'development_validation_bank.npz'
    assert sha(path) == json.loads((O/'BANK_IDENTITY.json').read_text())['sha256']
    bank = np.load(path)
    assert bank['truth'].shape == (2000,4) and bank['noise'].shape == (2000,30)
    assert bank['contrasts'].shape == (1000000,4) and bank['routing_uniforms'].shape == (2000,30)
    assert c['seed'] != 202611099206
    return bank


@torch.no_grad()
def evaluate(actor, out, label, deadline):
    out.mkdir(parents=True, exist_ok=False)
    bank = load_bank()
    lib = n.load_library()
    scorer_file = n.BASE/'source/frozen_evaluator.py'
    assert sha(scorer_file) == cfg()['evaluation']['scorer_sha256']
    scorer = load_module('categorical_pilot_scorer', scorer_file).contrast_scores
    contrasts = torch.tensor(bank['contrasts'], device='cuda')
    scores, routing, selected = [], [], []
    likelihood_calls = 0
    forward_seconds = scoring_seconds = 0.
    start = time.monotonic()
    actor.eval()
    for first in range(0,2000,16):
        engine.guard(deadline)
        theta = torch.tensor(bank['truth'][first:first+16], device='cuda')
        noise = torch.tensor(bank['noise'][first:first+16], device='cuda')
        uniforms = torch.tensor(bank['routing_uniforms'][first:first+16], device='cuda', dtype=torch.float32)
        ds = torch.zeros(len(theta),30,2,device='cuda')
        ys = torch.zeros(len(theta),30,device='cuda',dtype=torch.float64)
        gg, kk = [], []
        for t in range(30):
            x = lib.packed(ds,ys,t)
            torch.cuda.synchronize()
            tick = time.monotonic()
            if isinstance(actor,DecisionMoE):
                action, info = actor(x,uniform=uniforms[:,t],collect=True)
                proposals, idx = info['proposals'], info['selected']
                assert torch.equal(action,proposals[torch.arange(len(action),device='cuda'),idx])
                if actor.arm == 'uniform': assert torch.equal(info['gates'],torch.full_like(info['gates'],.25))
                gg.append(info['gates'].cpu().numpy());kk.append(idx.cpu().numpy())
            else: action = actor(x)
            torch.cuda.synchronize()
            forward_seconds += time.monotonic()-tick
            assert torch.isfinite(action).all() and (action.abs()<=4.000001).all()
            ds[:,t] = action
            ys[:,t] = signal(theta.reshape(-1,2,2),action.double())+.5*noise[:,t]
        torch.cuda.synchronize();tick=time.monotonic()
        bench=type('Benchmark',(),{'sigma':.5,'calls':0})()
        values,_=scorer(bench,dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=theta),contrasts,16384,'cuda')
        likelihood_calls += int(bench.calls)
        torch.cuda.synchronize();scoring_seconds+=time.monotonic()-tick
        scores.extend(values[:,-1].numpy())
        if gg:routing.append(np.stack(gg,1));selected.append(np.stack(kk,1))
        put(out/'STATUS.json',dict(status='EVALUATING',method=label,completed=first+len(theta),total=2000))
    score=np.asarray(scores)
    assert len(score)==2000 and np.isfinite(score).all()
    np.save(out/'scores.npy',score)
    d=dict(status='COMPLETE',method=label,seed=101,**engine.summaries(score),seconds=time.monotonic()-start,
        actor_batch16_seconds=forward_seconds,actor_amortized_seconds_per_design=forward_seconds/60000,
        scorer_seconds=scoring_seconds,simulator_observations=60000,negative_likelihood_calls=likelihood_calls,bank_sha256=sha(O/'development_validation_bank.npz'),
        evaluation=cfg()['evaluation'],all_full_episode_actions_valid=True)
    if routing:
        g=np.concatenate(routing);k=np.concatenate(selected)
        np.savez_compressed(out/'routing.npz',gates=g,selected=k)
        d.update(stage_mean_probabilities=g.mean(0).tolist(),stage_selection_fractions=[(k==j).mean(0).tolist() for j in range(4)])
    put(out/'RESULTS.json',d)
    return d


def models_from_common(parent,device):
    inference,experts=initial_models(parent)
    for k,e in enumerate(experts):
        cp=torch.load(O/f'pretrain_E{k+1}/policy.pt',map_location=device,weights_only=True)
        e.load_state_dict(cp['expert_state'])
    return inference,experts


def worker(task,index,arm,seconds):
    start=time.monotonic();deadline=start+seconds-30
    out=O/(f'pretrain_E{index+1}' if task=='pretrain' else arm)
    out.mkdir()
    put(out/'CONFIG.json',dict(task=task,expert=index+1 if task=='pretrain' else None,arm=arm,
        seed=101,protocol=cfg(),protocol_sha256=sha(C),source_manifest=json.loads((O/'SOURCE_MANIFEST.json').read_text())))
    setup()
    with lease(),engine.Resources(out):
        try:
            assert all(sha(M/k)==v for k,v in json.loads((O/'SOURCE_MANIFEST.json').read_text()).items())
            state=load_parent('cuda');a=state['algorithm'];parent=a.actor_net
            if task=='pretrain':
                inference,experts=initial_models(parent)
                actor=StandalonePolicy(inference,experts[index]).cuda()
                del inference,experts
                steps=cfg()['expert_updates_each'];label=f'MoE-sBOED-E{index+1}'
            else:
                inference,experts=models_from_common(parent,'cuda')
                seed(cfg()['router_initialization_seed'])
                actor=DecisionMoE(inference,experts,arm).cuda()
                del inference,experts
                steps=cfg()['mixture_updates'];label=NAMES[arm]
            configure_native(a,actor,task=='mixture',out)
            seed(101)
            put(out/'INITIALIZATION.json',dict(actor_digest=n.digest(actor.state_dict()),critic_digest=n.digest(a.critic_net.state_dict()),
                posterior_digest=n.digest({f'{t}.{i}.{k}':v for t,nets in a.post_approx.poi_post_nets.items() for i,net in enumerate(nets) for k,v in net.state_dict().items()}),
                parent_runtime_sha256=sha(P/'runtime.pkl'),fresh_initial_replay=10000,seed=101))
            COST.clear()
            receipt=train(a,out,steps,start,deadline-900)
            receipt['model_derived_calls']=dict(COST)
            put(out/'TRAINING.json',receipt)
            if task=='pretrain':
                torch.save(dict(expert_state=actor.expert.state_dict(),inference_state=actor.inference.state_dict(),role=ROLES[index]),out/'policy.pt')
            else:torch.save(dict(actor_state=actor.state_dict(),arm=arm,seed=101),out/'policy.pt')
            with (out/'runtime.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng(),task=task,arm=arm),f,protocol=5)
            actor.requires_grad_(False)
            evaluation=evaluate(actor,out/'evaluation',label,deadline)
            extras={}
            if task=='mixture' and arm=='learned':
                for k in range(4):
                    single=actor.standalone(k).cuda().eval().requires_grad_(False)
                    assert len([v for v in single.modules() if isinstance(v,PrivateExpert)])==1
                    extras[f'MoE-sBOED-E{k+1}']=evaluate(single,out/f'expert{k+1}_evaluation',f'MoE-sBOED-E{k+1}',deadline)
                    torch.save(single.state_dict(),out/f'expert{k+1}_policy.pt')
            assert sha(P/'runtime.pkl')==cfg()['parent_runtime_sha256'] and sha(P/'joint_actor.pt')==cfg()['parent_actor_sha256']
            assert all(sha(M/k)==v for k,v in json.loads((O/'SOURCE_MANIFEST.json').read_text()).items())
            put(out/'RESULTS.json',dict(status='COMPLETE',method=label,seed=101,training=receipt,evaluation=evaluation,
                standalone_evaluations=extras,seconds=time.monotonic()-start,source_preserved=True))
            put(out/'STATUS.json',dict(status='COMPLETE',method=label,seed=101,completed_updates=steps))
        except Exception as e:
            put(out/'STATUS.json',dict(status='FAILED_OR_CAPPED',error=repr(e),seconds=time.monotonic()-start));raise


def reference(seconds):
    start=time.monotonic();out=O/'collaborative_reference';out.mkdir()
    setup()
    with lease(),engine.Resources(out):
        state=load_parent('cuda');actor=state['algorithm'].actor_net
        cp=M/'experiments/location_finding/hec_v3_study_20261115/seed101_learned/joint_actor.pt'
        actor.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['actor_state'])
        actor.arm='learned';actor.correction_enabled=True;actor.encoders_enabled=True
        actor.gate_table=None;actor.configure('frozen');actor.eval()
        result=evaluate(actor,out/'evaluation','Existing collaborative MoE-sBOED',start+seconds-30)
        put(out/'RESULTS.json',dict(status='COMPLETE',evaluation=result,checkpoint=str(cp),checkpoint_sha256=sha(cp),training_updates=0,seconds=time.monotonic()-start))


def prepare():
    O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    path=O/'development_validation_bank.npz';assert not path.exists()
    rng=np.random.default_rng(cfg()['evaluation']['seed'])
    truth=rng.standard_normal((2000,4));noise=rng.standard_normal((2000,30));contrasts=rng.standard_normal((1000000,4))
    uniforms=np.random.default_rng(cfg()['evaluation']['routing_seed']).random((2000,30))
    np.savez(path,truth=truth,noise=noise,contrasts=contrasts,routing_uniforms=uniforms)
    put(O/'BANK_IDENTITY.json',dict(sha256=sha(path),protocol=cfg()['evaluation'],purpose='New development-validation bank;original benchmark test bank not opened or modified'))
    source={str(p.relative_to(M)):sha(p) for p in [*HERE.glob('*.py'),C,M/'tools/moe_sboed_categorical_pilot.py']}
    put(O/'SOURCE_MANIFEST.json',source)
    print(json.dumps(dict(bank=str(path),sha256=sha(path),out=str(O))))


def smoke():
    torch.set_num_threads(2)
    state=load_parent('cpu');parent=state['algorithm'].actor_net
    inference,experts=initial_models(parent)
    # Break symmetry only in a disposable test model to test categorical semantics.
    for k,e in enumerate(experts):
        with torch.no_grad():e.head[-1].bias.copy_(torch.tensor([-.6+.4*k,.5-.3*k]))
    seed(cfg()['router_initialization_seed']);learned=DecisionMoE(inference,experts,'learned')
    seed(cfg()['router_initialization_seed']);uniform=DecisionMoE(inference,experts,'uniform')
    assert n.digest(learned.state_dict())==n.digest(uniform.state_dict())
    rng=np.random.default_rng(20261116103);xx=[]
    for t in range(30):
        x=torch.zeros(2,117);x[:,t]=1
        x[:,30:30+2*t]=torch.tensor(rng.uniform(-4,4,(2,2*t)),dtype=torch.float32)
        x[:,88:88+t]=torch.tensor(rng.normal(size=(2,t)),dtype=torch.float32)
        xx.append(x)
    x=torch.cat(xx)
    with torch.no_grad():
        a,info=learned(x,uniform=torch.linspace(.01,.99,len(x)),collect=True)
        assert a.shape==(60,2) and (a.abs()<=4).all()
        assert torch.equal(a,info['proposals'][torch.arange(len(x)),info['selected']])
        assert torch.equal(torch.bincount(info['selected'],minlength=4),torch.full((4,),15))
        old_bias=learned.router[-1].bias.clone()
        learned.router[-1].bias.copy_(torch.tensor([.7,.1,.1,.1]).log())
        nonuniform,ni=learned(x,uniform=torch.linspace(.01,.99,len(x)),collect=True)
        assert torch.equal(torch.bincount(ni['selected'],minlength=4),torch.tensor([42,6,6,6]))
        assert torch.equal(nonuniform,ni['proposals'][torch.arange(len(x)),ni['selected']])
        learned.router[-1].bias.copy_(old_bias)
        for k in range(4):
            single=learned.standalone(k).eval();torch.testing.assert_close(single(x),info['proposals'][:,k],atol=0,rtol=0)
            assert sum(isinstance(v,PrivateExpert) for v in single.modules())==1
            assert not hasattr(single,'router')
        # No future-history leakage.
        future=x.clone()
        for row,t in enumerate(x[:,:30].argmax(-1).tolist()):
            future[row,30+2*t:88]=123.;future[row,88+t:]=456.
        torch.testing.assert_close(learned(future,uniform=torch.full((60,),.3)),learned(x,uniform=torch.full((60,),.3)),atol=0,rtol=0)
        uniform.router[-1].bias.copy_(torch.tensor([90.,-50.,10.,-10.]))
        _,ui=uniform(x,collect=True);assert torch.equal(ui['gates'],torch.full_like(ui['gates'],.25))
    # Exact-enumeration objective and gradient parity; nonlinear critic rejects action averaging.
    class Critic(nn.Module):
        def forward(self,v):return (v[...,-2:].square().sum(-1,keepdim=True)+.1*v[...,-2:-1])
    critic=Critic();small=x[:6]
    loss=learned.expected_actor_loss(small,critic)
    proposals,p=learned.proposals_and_probabilities(small)
    explicit=-sum((p[:,k]*critic(torch.cat([small,proposals[:,k]],-1)).squeeze(-1)).mean() for k in range(4))
    pars=tuple(p for p in learned.parameters() if p.requires_grad)
    ga=torch.autograd.grad(loss,pars,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(explicit,pars,allow_unused=True)
    for aa,bb in zip(ga,gb):
        assert (aa is None)==(bb is None)
        if aa is not None:torch.testing.assert_close(aa,bb,rtol=1e-4,atol=1e-6)
    averaged=critic(torch.cat([small,(p[...,None]*proposals).sum(1)],-1)).mean()
    assert abs(float(loss.detach()+averaged.detach()))>.01
    # Complete disposable full episodes,each standalone interface only.
    lib=n.load_library();theta=torch.tensor(rng.normal(size=(2,4)),dtype=torch.float64)
    with torch.no_grad():
        for k in range(4):
            single=learned.standalone(k);ds=torch.zeros(2,30,2);ys=torch.zeros(2,30,dtype=torch.float64)
            for t in range(30):
                design=single(lib.packed(ds,ys,t));assert torch.isfinite(design).all() and (design.abs()<=4).all()
                ds[:,t]=design;ys[:,t]=signal(theta.reshape(-1,2,2),design.double())
    # Validate actual native actor-branch patch and one disposable tiny update for both arms.
    a_native=state['algorithm'];native_steps=[]
    tiny_buffer={k:v[:8].clone() for k,v in a_native._runtime_buffer.items()}
    configure_native(a_native,copy.deepcopy(uniform),True,None)
    assert a_native.n_critic_update==5 and a_native.n_post_approx_update==5
    assert a_native.n_batch==10000 and a_native.n_buffer_init==10000
    a_native.n_newtraj=a_native.n_batch=4;a_native.n_buffer_init=8;a_native.n_buffer_max=16
    a_native.n_post_approx_update=a_native.n_critic_update=1;a_native.buffer_device='cpu'
    a_native._runtime_buffer=tiny_buffer
    a_native._runtime_p_max=torch.arange(1,17,dtype=torch.float32);a_native._runtime_p=[]
    class TinyLogger:
        def __init__(self):self.logger=self;self.tabular=engine.Table()
        def push_prefix(self,*args):pass
        def pop_prefix(self,*args):pass
        def log(self,*args):pass
        def dump_all(self):pass
    logger=TinyLogger();a_native.dowel=a_native.post_approx.dowel=logger;a_native.save_path=None
    profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile']
    tiny=dict(profile,discount=1.,critic_updates=1,posterior_updates=1,new_trajectories=4,replay_batch=4,initial_replay=8,replay_capacity=16)
    for arm in ['uniform','learned']:
        actor=copy.deepcopy(uniform if arm=='uniform' else learned)
        actor.configure();a_native.actor_net=actor;a_native.actor_target_net=copy.deepcopy(actor).requires_grad_(False)
        a_native.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=cfg()['actor_lr'])
        a_native.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a_native.actor_optimizer,gamma=.9999)
        patch_native_expectation(a_native,None)
        base=int(a_native.update);before=n.digest(actor.state_dict())
        n.original.train_native(a_native,logger,tiny,target=base+1)
        assert int(a_native.update)==base+1 and n.digest(actor.state_dict())!=before
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters() if p.requires_grad)
        assert all(p.grad is None for p in actor.router.parameters()) if arm=='uniform' else any(p.grad is not None for p in actor.router.parameters())
        native_steps.append(dict(arm=arm,updates=1,disposable_reduced_fixture=True))
    result=dict(status='PASS',device='CPU',all30_stages=True,private_heads_disjoint=True,
        all_four_standalone_full_episodes_valid=True,no_future_leakage=True,physical_action_is_exactly_one_proposal=True,
        uniform_weights_exact=True,exact_categorical_expectation_gradient_parity=True,not_Q_of_averaged_action=True,
        disposable_native_updates=native_steps,primary_test_bank_read=False)
    put(O/'SMOKE_CPU.json',result);print(json.dumps(result,indent=2))


def report():
    rows=[]
    for arm in ['learned','uniform']:
        path=O/arm/'RESULTS.json'
        if path.exists():
            d=json.loads(path.read_text());rows.append(dict(method=NAMES[arm],EIG=d['evaluation']['mean'],MCSE=d['evaluation']['MCSE'],score_path=str(O/arm/'evaluation/scores.npy')))
            if arm=='learned':
                for k in range(4):
                    e=d['standalone_evaluations'][f'MoE-sBOED-E{k+1}'];rows.append(dict(method=f'MoE-sBOED-E{k+1}',EIG=e['mean'],MCSE=e['MCSE'],score_path=str(O/arm/f'expert{k+1}_evaluation/scores.npy')))
    reference=O/'collaborative_reference/RESULTS.json'
    if reference.exists():
        e=json.loads(reference.read_text())['evaluation'];rows.append(dict(method='Existing collaborative MoE-sBOED',EIG=e['mean'],MCSE=e['MCSE'],score_path=str(O/'collaborative_reference/evaluation/scores.npy')))
    lookup={r['method']:r for r in rows};comparisons={};expand=False
    if all(k in lookup for k in ['MoE-sBOED','MoE-sBOED-uniform','Existing collaborative MoE-sBOED']):
        initial=[json.loads((O/arm/'INITIALIZATION.json').read_text()) for arm in ['learned','uniform']]
        assert initial[0]['actor_digest']==initial[1]['actor_digest'] and initial[0]['critic_digest']==initial[1]['critic_digest'] and initial[0]['posterior_digest']==initial[1]['posterior_digest']
        a=np.load(lookup['MoE-sBOED']['score_path'])
        for name,row in lookup.items():
            if name!='MoE-sBOED':comparisons[name]=engine.summaries(a-np.load(row['score_path']))
        expand=(comparisons['MoE-sBOED-uniform']['CI95'][0]>0 and comparisons['Existing collaborative MoE-sBOED']['CI95'][0]>-cfg()['competitive_margin_nats'])
    result=dict(status='COMPLETE' if len(rows)==7 else 'PENDING',seed=101,rows=rows,paired_comparisons=comparisons,
        expand_to_five_seeds=expand if len(rows)==7 else None,bank_identity=json.loads((O/'BANK_IDENTITY.json').read_text()),
        uncertainty='Paired case uncertainty for one continuation seed;not across-training-seed robustness.',
        compute='Each method charged all4x200 shared standalone updates plus1200 independently run mixture updates;2.05M episodes including fresh replay per method. Shared pretraining and validation wall charged to each4h cap;actual timings in receipts.')
    put(R/'RESULTS.json',result)
    text='# Categorical private-head pilot,seed101\n\n| Method | EIG (nats) | MCSE |\n|---|---:|---:|\n'
    for row in rows:text+=f"| {row['method']} | {row['EIG']:.5f} | {row['MCSE']:.5f} |\n"
    text+='\nPaired comparisons:\n\n```json\n'+json.dumps(comparisons,indent=2)+'\n```\n\n'
    text+='Expansion decision:'+('YES:predeclared pilot criteria met;five-seed validation still required.' if expand else 'NO expansion authorized automatically;pilot pending or criteria not met.')+'\n\n'+result['uncertainty']+'\n\n'+result['compute']+'\n'
    (R/'RESULTS.md').write_text(text)


def controller():
    O.mkdir(exist_ok=True,parents=True)
    lock=(O/'.controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((O/'SMOKE_CPU.json').read_text())['status']=='PASS'
    assert not (O/'LAUNCH.json').exists(),'Existing controller attempt preserved;no duplicate restart'
    start=time.monotonic();put(O/'LAUNCH.json',dict(pid=os.getpid(),started=time.time(),protocol_sha256=sha(C)))
    def execute(args,log,seconds):
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB',next=args));time.sleep(20)
        assert seconds>120
        with (O/log).open('x') as f:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(int(seconds-30)),sys.executable,'-u',str(M/'tools/moe_sboed_categorical_pilot.py'),*args,'--seconds',str(seconds)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),worker_pid=child.pid,args=args,started=time.time()))
            code=child.wait()
        if code:raise RuntimeError(f'Preserved failed/capped component:{args};returncode={code}')
    try:
        for k in range(4):
            remaining=cfg()['common_pretraining_hard_seconds']-(time.monotonic()-start)
            assert not (O/f'pretrain_E{k+1}').exists()
            put(O/'STATUS.json',dict(status='PRETRAINING_STANDALONE_EXPERT',expert=k+1,seed=101))
            execute(['--task','pretrain','--index',str(k)],f'pretrain_E{k+1}.log',remaining)
        execute(['--task','reference'],'collaborative_reference.log',1800)
        common=time.monotonic()-start
        assert common<cfg()['hard_total_seconds_per_method']-1800
        for k in range(4):assert json.loads((O/f'pretrain_E{k+1}/RESULTS.json').read_text())['evaluation']['all_full_episode_actions_valid']
        put(O/'COMMON_PRETRAINING.json',dict(status='COMPLETE',seconds_charged_to_each_method=common,
            expert_updates=4*cfg()['expert_updates_each'],simulator_episodes=4*(10000+1000*cfg()['expert_updates_each']),
            full_episode_EIG_verified_before_router_training=True))
        for arm in ['learned','uniform']:
            method_start=time.monotonic();remaining=cfg()['hard_total_seconds_per_method']-common
            put(O/'STATUS.json',dict(status='TRAINING_MIXTURE',method=NAMES[arm],seed=101,pending=['uniform'] if arm=='learned' else []))
            execute(['--task','mixture','--arm',arm],arm+'.log',remaining)
            assert common+time.monotonic()-method_start<=cfg()['hard_total_seconds_per_method']
            report()
        report();put(O/'STATUS.json',dict(status='COMPLETE',seed=101,report=str(R/'RESULTS.md')))
    except Exception as e:
        put(O/'STATUS.json',dict(status='FAILED_OR_CAPPED',error=repr(e),seconds=time.monotonic()-start));report();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['prepare','smoke','controller','pretrain','mixture','reference','report'],required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--arm',choices=['learned','uniform'],default='learned');p.add_argument('--seconds',type=float,default=14400)
    a=p.parse_args()
    if a.task=='prepare':prepare()
    elif a.task=='smoke':smoke()
    elif a.task=='controller':controller()
    elif a.task=='reference':reference(a.seconds)
    elif a.task=='report':report()
    else:worker(a.task,a.index,a.arm,a.seconds)


if __name__=='__main__':main()
