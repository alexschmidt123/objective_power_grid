"""Seed101 uniform-initialized routing trial; frozen scientific dependencies.

No test-bank selection. Native variational actor/critic/posterior updates only.
The uniform control is trained independently and never uses learned gates.
"""
import argparse, copy, fcntl, hashlib, io, json, os, pathlib, subprocess, sys, time

M = pathlib.Path.home() / 'Documents/objective_power_grid'
S = M / 'src/research/hec_v3_study'
C = M / 'configs/moe_sboed_uniform_residual_seed101_20261116.json'
O = M / 'experiments/location_finding/moe_sboed_uniform_residual_seed101_20261116'
P = M / 'experiments/location_finding/hec_v3_study_20261115/seed101_uniform'
sys.dont_write_bytecode = True
sys.path.insert(0, str(S))
import engine
from runtime import n, setup, lease, sha, put, signal, load_module, LOCK
from model import StudyMoE
from frozen_feature_policy import FeatureMoE
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

NAMES = {'learned': 'MoE-sBOED', 'uniform': 'MoE-sBOED-uniform'}


class UniformResidualMoE(StudyMoE):
    """Same experts/fusion; only the top-level gate equation is replaced."""
    def configure(self, phase):
        if not hasattr(self, 'trial_arm'):
            return super().configure(phase)
        super().configure('frozen' if phase in ['warmup', 'frozen'] else 'joint')
        self.prior_coordinator.requires_grad_(False)
        self.prior_response_gate.requires_grad_(False)
        self.router.requires_grad_(phase != 'frozen' and self.trial_arm == 'learned')
        for p in self.parameters():
            p.grad = None

    def compute(self, x, mode='learned', expert=None, collect=False):
        if mode == 'expert':
            return super().compute(x, mode, expert, collect)
        c = self.context(x)
        packets = [e(c) for e in self.experts]
        first = torch.stack([p['first'] for p in packets], 1)
        response = torch.stack([p['response'] for p in packets], 1)
        innovation = torch.stack([p['innovation'] for p in packets], 1)
        projected = torch.stack([
            F.linear(p['first'], self.decoder.weight[:, 128+64*k:128+64*(k+1)])
            + F.linear(p['response'], self.decoder.response_weight[:, 64*k:64*(k+1)])
            + p['innovation'] for k, p in enumerate(packets)], 1)
        proposals = torch.stack([self.decode_one(c, k, p) for k, p in enumerate(packets)], 1)
        remaining = (30-c['stage']).to(x.dtype)[:, None]/30
        norms = torch.log1p(projected.square().mean(-1))
        offsets = (proposals-c['a0'][:, None])/4
        distances = torch.stack([(proposals[:, i]-proposals[:, j]).norm(dim=-1)/8
                                 for i in range(4) for j in range(i+1, 4)], -1)
        inp = torch.cat([c['s'], remaining, norms, offsets.flatten(1), distances], -1).detach()
        if self.trial_arm == 'uniform' or mode == 'uniform':
            gates = x.new_full((len(x), 4), .25)
            assert not gates.requires_grad and torch.equal(gates, torch.full_like(gates, .25))
        else:
            gates = self.router(inp).softmax(-1)
        if self.diagnostic_gate is not None:
            assert not self.training, 'Diagnostic overrides forbidden during training'
            gates = self.diagnostic_gate.to(x).expand(len(x), 4)
        action = self.decoder(c, gates, first, response, innovation)
        if not collect:
            return action
        return action, dict(gates=gates, history=c['s'], commands=c['commands'],
                            first=first, responses=response, innovation=innovation,
                            projected=projected, proposals=proposals, router_input=inp, scale=c['scale'])


def config():
    return json.loads(C.read_text())


def load_runtime(device):
    cfg = config()
    assert sha(P/'runtime.pkl') == cfg['parent_runtime_sha256']
    assert sha(P/'joint_actor.pt') == cfg['parent_actor_sha256']
    old = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
    try:
        with (P/'runtime.pkl').open('rb') as f:
            state = n.cloudpickle.load(f)
    finally:
        torch.storage._load_from_bytes = old
    a = state['algorithm']
    assert a.update == 18001 and a.discount == 1 and not a.use_PCE
    cp = torch.load(P/'joint_actor.pt', map_location=device, weights_only=True)['actor_state']
    assert all(torch.equal(v, cp[k]) for k, v in a.actor_net.state_dict().items())
    for op, _ in engine.opts(a):
        op._optimizer_step_pre_hooks.clear()
        op._optimizer_step_post_hooks.clear()
    return state


def convert(actor, arm):
    actor = copy.deepcopy(actor)
    actor.__class__ = UniformResidualMoE
    actor.trial_arm = arm
    actor.arm = arm
    actor.diagnostic_gate = None
    actor.gate_table = None
    actor.observation_swap = False
    actor.swap_gates = False
    actor.remove_feature = None
    actor.encoders_enabled = True
    nn.init.zeros_(actor.router[-1].weight)
    nn.init.zeros_(actor.router[-1].bias)
    actor.configure('frozen')
    return actor


def group_norms(actor):
    values = {}
    for name, p in actor.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            group = '.'.join(name.split('.')[:2])
            values[group] = values.get(group, 0.) + float(p.grad.square().sum())
        else:
            assert p.grad is None, name
    return {k: v**.5 for k, v in values.items()}


class SmokeLogger:
    def __init__(self):
        self.logger = self
        self.tabular = engine.Table()
    def push_prefix(self, *args): pass
    def pop_prefix(self, *args): pass
    def log(self, *args): pass
    def dump_all(self): pass


class Logger:
    def __init__(self, a, out, phase, steps, start, deadline):
        self.a, self.out, self.phase, self.steps = a, out, phase, steps
        self.base = int(a.update)
        self.start, self.deadline = start, deadline
        self.logger = self
        self.tabular = engine.Table()
        self.counts = dict(actor=0, critic=0, posterior=0)
        self.gradients = {}

    def push_prefix(self, *args): pass
    def pop_prefix(self, *args): pass
    def log(self, *args): pass

    def dump_all(self):
        done = int(self.a.update)-self.base
        assert self.counts == dict(actor=done, critic=5*done, posterior=150*done)
        if done == 1 or done % 25 == 0 or done == self.steps:
            row = dict(status='TRAINING', method=NAMES[self.a.actor_net.trial_arm], seed=101,
                       phase=self.phase, phase_updates=done, phase_target=self.steps,
                       completed_updates=int(self.a.update)-18001, counts=self.counts.copy(),
                       seconds=time.monotonic()-self.start, gradient_group_norms=self.gradients,
                       actor_lr=self.a.actor_optimizer.param_groups[0]['lr'])
            with (self.out/'TRAINING_METRICS.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            put(self.out/'STATUS.json', row)
            print(json.dumps(row), flush=True)
        engine.guard(self.deadline)


def train_phase(a, out, phase, steps, start, deadline):
    a.actor_net.configure(phase)
    logger = Logger(a, out, phase, steps, start, deadline)
    a.dowel = logger
    a.post_approx.dowel = logger
    a.save_path = None
    for op, key in engine.opts(a):
        op._optimizer_step_pre_hooks.clear()
        op._optimizer_step_post_hooks.clear()
        op.register_step_post_hook(lambda *args, key=key: logger.counts.__setitem__(key, logger.counts[key]+1))
    def check(*args):
        logger.gradients = group_norms(a.actor_net)
        assert a.actor_net.diagnostic_gate is None
        if a.actor_net.trial_arm == 'uniform':
            assert not any(p.requires_grad for p in a.actor_net.router.parameters())
    a.actor_optimizer.register_step_pre_hook(check)
    frozen = engine.frozen_digest(a.actor_net)
    profile = json.loads((n.P/'frozen_protocol.json').read_text())['profile']
    profile['discount'] = 1.
    tick = time.monotonic()
    n.original.train_native(a, logger, profile, target=logger.base+steps)
    assert engine.frozen_digest(a.actor_net) == frozen
    for op, _ in engine.opts(a):
        op._optimizer_step_pre_hooks.clear()
        op._optimizer_step_post_hooks.clear()
    receipt = dict(status='COMPLETE', updates=int(a.update)-logger.base, counts=logger.counts,
                   seconds=time.monotonic()-tick, frozen_state_preserved=True,
                   trainable_parameters=sum(p.numel() for p in a.actor_net.parameters() if p.requires_grad),
                   actor_updates_with_trainable_parameters=steps if any(p.requires_grad for p in a.actor_net.parameters()) else 0,
                   objective='Unmodified native full-horizon variational actor-critic EIG,use_PCE=False,discount1')
    torch.save(dict(actor_state=a.actor_net.state_dict(), native_iteration=int(a.update)), out/(phase+'_actor.pt'))
    put(out/(phase+'_TRAINING.json'), receipt)
    return receipt


@torch.no_grad()
def diagnostic(a, out, deadline):
    """One-step gate interventions, then identical uniform continuation.

    Paired on-policy joint samples, not noiseless conditional per-history Q.
    Fresh prior contrasts; never reads the primary evaluation bank.
    """
    cfg = config()['diagnostic']
    rng = np.random.default_rng(cfg['seed'])
    count = cfg['cases']
    truth = torch.tensor(rng.standard_normal((count, 4)), device='cuda')
    noise = torch.tensor(rng.standard_normal((count, 30)), device='cuda')
    contrasts = torch.tensor(rng.standard_normal((cfg['contrasts'], 4)), device='cuda')
    actor = a.actor_net.eval()
    lib = n.load_library()
    # The scorer is the same implementation, but uses ONLY fresh diagnostic draws.
    ds = torch.zeros(count, 30, 2, device='cuda')
    ys = torch.zeros(count, 30, device='cuda', dtype=torch.float64)
    prefixes = {}
    for t in range(30):
        if t in cfg['stages']:
            prefixes[t] = (ds.clone(), ys.clone())
        x = lib.packed(ds, ys, t)
        design = actor(x)
        ds[:, t] = design
        ys[:, t] = signal(truth.reshape(-1, 2, 2), design.double())+.5*noise[:, t]
    # Reuse canonical scorer function without loading its canonical test bank.
    scorer_path = n.BASE/'source/frozen_evaluator.py'
    assert sha(scorer_path) == LOCK['scorer_sha256']
    scorer = load_module('routing_diagnostic_frozen_scorer', scorer_path).contrast_scores
    rows = []
    arrays = {}
    weights = [('uniform', np.full(4, .25))]
    for k in range(4):
        d = np.full(4, -cfg['epsilon']/4)
        d[k] += cfg['epsilon']
        weights.extend([(f'plus_E{k+1}', .25+d), (f'minus_E{k+1}', .25-d)])
    from scipy.stats import t as student
    for stage, (base_ds, base_ys) in prefixes.items():
        scores, predictions = {}, {}
        for label, weight in weights:
            engine.guard(deadline)
            dd, yy = base_ds.clone(), base_ys.clone()
            x = lib.packed(dd, yy, stage)
            actor.diagnostic_gate = torch.tensor(weight, device='cuda', dtype=torch.float32)
            design = actor(x)
            predictions[label] = a.critic_net(torch.cat([x, design], -1)).squeeze(-1).cpu().numpy()
            actor.diagnostic_gate = None
            dd[:, stage] = design
            yy[:, stage] = signal(truth.reshape(-1, 2, 2), design.double())+.5*noise[:, stage]
            for t in range(stage+1, 30):
                x = lib.packed(dd, yy, t)
                design = actor(x)
                dd[:, t] = design
                yy[:, t] = signal(truth.reshape(-1, 2, 2), design.double())+.5*noise[:, t]
            bench = type('Benchmark', (), {'sigma': .5, 'calls': 0})()
            values, _ = scorer(bench, dict(actions=list(dd.unbind(1)), observations=list(yy[..., None].unbind(1)), truth=truth), contrasts, 4096, 'cuda')
            scores[label] = values[:, -1].numpy()
            arrays[f'stage{stage}_{label}'] = scores[label]
        for label, _ in weights[1:]:
            delta = scores[label]-scores['uniform']
            pred = predictions[label]-predictions['uniform']
            stat = engine.summaries(delta)
            q = float(student.ppf(1-.05/(len(cfg['stages'])*8), count-1))
            lower = float(delta.mean()-q*delta.std(ddof=1)/count**.5)
            rows.append(dict(stage=stage, intervention=label, **stat, simultaneous_one_sided_lower=lower,
                             critic_mean_delta=float(pred.mean()), sign_agreement_fraction=float(np.mean(np.sign(pred)==np.sign(delta))),
                             per_history_delta_is_single_sample=True))
        put(out/'DIAGNOSTIC_PROGRESS.json', dict(stage=stage, completed_stages=len(rows)//8))
    actor.diagnostic_gate = None
    np.savez_compressed(out/'diagnostic_scores.npz', **arrays)
    result = dict(status='COMPLETE', rows=rows, headroom_detected=any(r['simultaneous_one_sided_lower']>0 for r in rows),
                  protocol=cfg, scope='Fresh paired suffix simulations;one future per joint history/theta draw;limited power,finite-contrast estimator. Not conditional exact Q. No training-target replacement.',
                  affects_training_objective=False, affects_hyperparameters=False,
                  total_rollout_observations=count*(30+9*sum(30-t for t in cfg['stages'])))
    put(out/'ROUTING_HEADROOM_DIAGNOSTIC.json', result)
    return result


def smoke():
    torch.set_num_threads(2)
    state = load_runtime('cpu')
    parent = state['algorithm'].actor_net.eval()
    learned, uniform = [convert(parent, arm).eval() for arm in ['learned', 'uniform']]
    assert n.digest(learned.state_dict()) == n.digest(uniform.state_dict())
    saved = np.load(P/'uniform_routing.npz')['gates']
    assert saved.shape == (2000, 30, 4) and np.all(saved == .25)
    bank = np.random.default_rng(202611160311)
    xs = []
    for t in range(30):
        x = torch.zeros(2, 117)
        x[:, t] = 1
        x[:, 30:30+2*t] = torch.tensor(bank.uniform(-4, 4, (2, 2*t)), dtype=torch.float32)
        x[:, 88:88+t] = torch.tensor(bank.normal(0, 1, (2, t)), dtype=torch.float32)
        xs.append(x)
    x = torch.cat(xs)
    with torch.no_grad():
        au, iu = uniform(x, collect=True)
        al, il = learned(x, collect=True)
        original = parent(x)
        torch.testing.assert_close(au, original, atol=0, rtol=0)
        torch.testing.assert_close(al, au, atol=0, rtol=0)
        assert torch.equal(iu['gates'], torch.full_like(iu['gates'], .25))
        uniform.router[-1].weight.fill_(10)
        uniform.router[-1].bias.copy_(torch.tensor([-50., -10., 10., 50.]))
        changed, ic = uniform(x, collect=True)
        torch.testing.assert_close(changed, au, atol=0, rtol=0)
        assert torch.equal(ic['gates'], iu['gates'])
    learned.train().configure('warmup')
    crit = state['algorithm'].critic_net
    loss = -crit(torch.cat([x, learned(x)], -1)).mean()
    loss.backward()
    norms = group_norms(learned)
    assert norms['router.2'] > 0
    # Native warmup control has no trainable actor; loss still differentiates critic.
    uniform.train().configure('warmup')
    assert not any(p.requires_grad for p in uniform.parameters())
    op = torch.optim.Adam(uniform.parameters(), lr=.0001)
    before = n.digest(uniform.state_dict())
    op.zero_grad()
    (-crit(torch.cat([x, uniform(x)], -1)).mean()).backward()
    op.step()
    assert before == n.digest(uniform.state_dict())
    assert not group_norms(uniform)
    counts = {}
    for arm, model in [('learned', learned), ('uniform', uniform)]:
        counts[arm] = {}
        for phase in ['warmup', 'joint']:
            model.configure(phase)
            counts[arm][phase] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    native_smokes = []
    a = state['algorithm']
    # Disposable reduced replay fixture. No saved checkpoint is modified.
    a.n_newtraj = a.n_batch = 4
    a.n_buffer_init, a.n_buffer_max = 8, 16
    a.n_post_approx_update, a.n_critic_update = 1, 1
    a.buffer_device = 'cpu'
    a._runtime_buffer = {k: v[:8].clone() for k, v in a._runtime_buffer.items()}
    a._runtime_p_max = torch.arange(1, 17, dtype=torch.float32)
    a._runtime_p = []
    logger = SmokeLogger()
    a.dowel = a.post_approx.dowel = logger
    a.save_path = None
    profile = json.loads((n.P/'frozen_protocol.json').read_text())['profile']
    tiny = dict(profile, discount=1., critic_updates=1, posterior_updates=1,
                new_trajectories=4, replay_batch=4, initial_replay=8, replay_capacity=16)
    for arm in ['uniform', 'learned']:
        model = convert(parent, arm).train()
        model.configure('warmup')
        a.actor_net = model
        a.actor_target_net = copy.deepcopy(model)
        a.actor_target_net.configure('frozen')
        a.actor_optimizer = torch.optim.Adam(model.parameters(), lr=config()['actor_lr'])
        a.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer, gamma=.9999)
        old_digest, old_update = n.digest(model.state_dict()), int(a.update)
        n.original.train_native(a, logger, tiny, target=old_update+1)
        assert int(a.update) == old_update+1
        changed = n.digest(model.state_dict()) != old_digest
        assert changed == (arm == 'learned')
        native_smokes.append(dict(method=NAMES[arm], native_updates=1, actor_changed=changed,
                                  reduced_fixture_only=True))
    result = dict(status='PASS', device='CPU', stages=30, histories=60,
                  initial_learned_uniform_parent_actions_bitwise_equal=True,
                  uniform_immune_to_router_weight_perturbation=True,
                  uniform_warmup_no_parameter_change=True, native_critic_loss_router_gradient=norms,
                  trainable_actor_parameters=counts, canonical_test_bank_loaded=False)
    result['disposable_native_updates'] = native_smokes
    print(json.dumps(result, indent=2))


def worker(arm):
    start = time.monotonic()
    out = O/arm
    out.mkdir()
    cfg = config()
    put(out/'CONFIG.json', dict(method=NAMES[arm], seed=101, protocol=cfg, protocol_sha256=sha(C)))
    setup()
    with lease(), engine.Resources(out):
        try:
            state = load_runtime('cuda')
            a = state['algorithm']
            actor = convert(a.actor_net, arm).cuda()
            if arm == 'uniform':
                # Constant throughout; no intervention/table mechanism exists in this subclass.
                assert actor.trial_arm == 'uniform'
            a.actor_net = actor
            a.actor_target_net = copy.deepcopy(actor)
            a.actor_target_net.configure('frozen')
            a.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg['actor_lr'])
            a.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer, gamma=.9999)
            engine.seed_all(101)
            put(out/'INITIALIZATION.json', dict(method=NAMES[arm], seed=101, parent_update=int(a.update),
                actor_digest=n.digest(actor.state_dict()), critic_digest=n.digest(a.critic_net.state_dict()),
                parent_runtime_sha256=sha(P/'runtime.pkl'), actor_lr=cfg['actor_lr'],
                exact_uniform_initialization=True, protocol_sha256=sha(C)))
            # Fixed predeclared trial: diagnosis is not checkpoint or hyperparameter selection.
            if arm == 'learned':
                diagnostic(a, out, start+cfg['diagnostic']['hard_seconds'])
                engine.seed_all(101)
            actor.train()
            phases = [train_phase(a, out, phase, steps, start, start+cfg['training_soft_seconds'])
                      for phase, steps in [('warmup', 200), ('joint', 1800)]]
            torch.save(dict(actor_state=actor.state_dict(), actor_target_state=a.actor_target_net.state_dict(),
                            native_iteration=int(a.update), method=NAMES[arm], seed=101), out/'final_actor.pt')
            for op, _ in engine.opts(a):
                op._optimizer_step_pre_hooks.clear()
                op._optimizer_step_post_hooks.clear()
            with (out/'runtime.pkl').open('wb') as f:
                n.cloudpickle.dump(dict(algorithm=a, rng=n.original.rng(), method=NAMES[arm], seed=101), f, protocol=5)
            actor.configure('frozen')
            actor.eval()
            assert actor.diagnostic_gate is None
            scores, ev = engine.evaluate(actor, out, arm, start+14100, collect=True)
            if arm == 'uniform':
                g = np.load(out/'uniform_routing.npz')['gates']
                assert g.shape == (2000, 30, 4) and np.all(g == .25)
            assert sha(P/'runtime.pkl') == cfg['parent_runtime_sha256'] and sha(P/'joint_actor.pt') == cfg['parent_actor_sha256']
            put(out/'RESULTS.json', dict(status='COMPLETE', method=NAMES[arm], seed=101, phases=phases,
                updates=2000, actor_updates_with_trainable_parameters=2000 if arm=='learned' else 1800,
                new_training_episodes=2000000, simulator_training_observations=60000000,
                training_seconds=sum(p['seconds'] for p in phases), total_seconds=time.monotonic()-start,
                evaluation=ev, historical_parent_preserved=True, final_actor_sha256=sha(out/'final_actor.pt'),
                limitation='One additional continuation experiment,not five-seed validation or from-scratch compute parity. Uniform warmup actor steps are no-ops;critic/posterior updates remain matched.'))
            put(out/'STATUS.json', dict(status='COMPLETE', method=NAMES[arm], seed=101, completed_updates=2000))
        except Exception as e:
            put(out/'STATUS.json', dict(status='FAILED_OR_CAPPED', method=NAMES[arm], seed=101, error=repr(e)))
            raise


def report():
    rows = []
    for arm in ['learned', 'uniform']:
        p = O/arm/'RESULTS.json'
        if p.exists():
            d = json.loads(p.read_text())
            if d['status'] == 'COMPLETE':
                rows.append(dict(method=NAMES[arm], seed=101, EIG=d['evaluation']['mean'], MCSE=d['evaluation']['MCSE']))
    d = dict(status='COMPLETE' if len(rows)==2 else 'PENDING', rows=rows, seed=101)
    if len(rows) == 2:
        aa, bb = [json.loads((O/arm/'INITIALIZATION.json').read_text()) for arm in ['learned', 'uniform']]
        assert aa['actor_digest'] == bb['actor_digest'] and aa['critic_digest'] == bb['critic_digest']
        for arm in ['learned', 'uniform']:
            assert json.loads((O/arm/'RESULTS.json').read_text())['evaluation']['protocol'] == LOCK
        delta = np.load(O/'learned/learned_scores.npy')-np.load(O/'uniform/uniform_scores.npy')
        d['paired_learned_minus_uniform'] = engine.summaries(delta)
        d['uncertainty_scope'] = 'Paired case uncertainty conditional on one training seed and fixed contrast bank;not across-seed robustness.'
    put(O/'COMPARISON.json', d)
    text = '# Seed101 uniform-initialized routing trial\n\n| Method | Seed | Full-horizon EIG (nats) | MCSE |\n|---|---:|---:|---:|\n'
    for r in rows:
        text += f"| {r['method']} | 101 | {r['EIG']:.6f} | {r['MCSE']:.6f} |\n"
    if len(rows) < 2:
        text += '\nPending: no superiority claim.\n'
    else:
        text += '\nPaired difference and95% CI: '+json.dumps(d['paired_learned_minus_uniform'])+'\n'
    (O/'RESULTS.md').write_text(text)


def controller():
    O.mkdir(exist_ok=True, parents=True)
    lock = (O/'.controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = config()
    assert json.loads((O/'SMOKE_CPU.json').read_text())['status'] == 'PASS'
    put(O/'LAUNCH.json', dict(pid=os.getpid(), started=time.time(), protocol_sha256=sha(C), queue=cfg['queue']))
    failed = []
    for arm in cfg['queue']:
        result = O/arm/'RESULTS.json'
        if result.exists() and json.loads(result.read_text())['status'] == 'COMPLETE':
            continue
        if (O/arm).exists():
            failed.append(dict(arm=arm, reason='PRESERVED_EXISTING_NO_AUTORETRY'))
            continue
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            put(O/'STATUS.json', dict(status='WAITING_FOR_UNRELATED_GPU_JOB', method=NAMES[arm]))
            time.sleep(20)
        with (O/(arm+'.log')).open('x') as f:
            child = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=30', '14370', sys.executable, '-u', str(pathlib.Path(__file__).resolve()), '--worker', arm],
                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
            put(O/'ACTIVE_PROCESS.json', dict(controller_pid=os.getpid(), worker_pid=child.pid, arm=arm, method=NAMES[arm], seed=101, started=time.time()))
            put(O/'STATUS.json', dict(status='RUNNING', method=NAMES[arm], seed=101, pending=cfg['queue'][cfg['queue'].index(arm)+1:]))
            code = child.wait()
            if code:
                failed.append(dict(arm=arm, returncode=code))
        report()
    report()
    put(O/'STATUS.json', dict(status='COMPLETE' if not failed else 'FINISHED_WITH_FAILURES', failed=failed))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--worker', choices=['learned', 'uniform'])
    p.add_argument('--controller', action='store_true')
    p.add_argument('--report', action='store_true')
    args = p.parse_args()
    if args.smoke: smoke()
    elif args.worker: worker(args.worker)
    elif args.controller: controller()
    elif args.report: report()
    else: p.error('Choose a task')
