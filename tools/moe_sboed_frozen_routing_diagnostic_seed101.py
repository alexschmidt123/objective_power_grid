"""Frozen uniform101 counterfactual routing; no training and no primary-bank reads."""
import argparse, copy, importlib.util, json, math, pathlib, sys, time
M = pathlib.Path.home()/'Documents/objective_power_grid'
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('uniform_residual_trial', M/'tools/moe_sboed_uniform_residual_seed101.py')
trial = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trial
spec.loader.exec_module(trial)
from runtime import n, sha, put, setup, lease, signal, load_module, LOCK
from hec_policy import mean_signal
import torch
import numpy as np
from scipy.stats import t as student, spearmanr

O = M/'experiments/location_finding/moe_sboed_uniform_residual_seed101_20261116/frozen_diagnostic'
C = M/'configs/moe_sboed_frozen_routing_diagnostic_seed101_20261116.json'


def summary(v, simultaneous=1):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return dict(n=len(v), mean=float(v.mean()), SE=None, CI95=None)
    se = float(v.std(ddof=1)/np.sqrt(len(v)))
    q = float(student.ppf(1-.025/simultaneous, len(v)-1))
    return dict(n=len(v), mean=float(v.mean()), SE=se,
                CI95=[float(v.mean()-q*se), float(v.mean()+q*se)])


@torch.no_grad()
def conditional_latents(actor, x, ds, ys, stage, cfg, count):
    if stage == 0:
        return torch.randn(count, 4, device=x.device, dtype=torch.float64), dict(exact_prior=True, ESS=None, adequate=True)
    w, mu, sd = actor.context.belief.params(x)
    w, mu, sd = w[0].double(), mu[0].double(), sd[0].double()
    candidates, logweights = [], []
    size = 0
    target = cfg['initial_importance_candidates']
    while True:
        extra = target-size
        component = torch.multinomial(w, extra, replacement=True)
        theta = mu[component]+sd[component]*torch.randn(extra, 4, device=x.device, dtype=torch.float64)
        prior_mask = torch.rand(extra, device=x.device) < cfg['prior_proposal_fraction']
        theta[prior_mask] = torch.randn(int(prior_mask.sum()), 4, device=x.device, dtype=torch.float64)
        lp_prior = -.5*theta.square().sum(-1)-2*math.log(2*math.pi)
        lp_gmm = torch.logsumexp(w.clamp_min(1e-300).log()[None]
                    + torch.distributions.Normal(mu[None], sd[None]).log_prob(theta[:, None]).sum(-1), -1)
        alpha = cfg['prior_proposal_fraction']
        logq = torch.logaddexp(lp_gmm+math.log1p(-alpha), lp_prior+math.log(alpha))
        pred = mean_signal(theta[None], ds[:stage][None].double())[0].transpose(0, 1)
        ll = -2*(pred-ys[:stage].double()[None]).square().sum(-1)
        candidates.append(theta)
        logweights.append(ll+lp_prior-logq)
        size = target
        all_w = torch.cat(logweights).softmax(0)
        ess = float(all_w.square().sum().reciprocal())
        if ess >= cfg['minimum_importance_ESS'] or target == cfg['maximum_importance_candidates']:
            break
        target = min(target*4, cfg['maximum_importance_candidates'])
    theta = torch.cat(candidates)
    idx = torch.multinomial(all_w, count, replacement=True)
    return theta[idx], dict(exact_prior=False, ESS=ess, candidates=size,
                           maximum_normalized_weight=float(all_w.max()),
                           adequate=ess >= cfg['minimum_importance_ESS'],
                           approximation='Self-normalized importance resampling from frozen GMM/prior mixture,corrected using exact observed-prefix likelihood. Finite-pool uncertainty is not covered by rollout CIs.')


@torch.no_grad()
def run(smoke=False):
    cfg = json.loads(C.read_text())
    if smoke:
        cfg = dict(cfg, histories_per_stage=1, stages=[0, 9], suffix_replicates=8,
                   contrasts=128, initial_importance_candidates=256, maximum_importance_candidates=1024,
                   minimum_importance_ESS=2)
        out = O/'smoke'
    else:
        out = O/'full'
        assert json.loads((O/'smoke/RESULTS.json').read_text())['status'] == 'PASS'
    out.mkdir(parents=True)
    setup()
    start = time.monotonic()
    deadline = start+cfg['hard_seconds']-60
    with lease(), trial.engine.Resources(out):
        state = trial.load_runtime('cuda')
        a = state['algorithm']
        actor = trial.convert(a.actor_net, 'uniform').cuda().eval()
        actor.requires_grad_(False)
        assert not any(p.requires_grad for p in actor.parameters())
        original_router = copy.deepcopy(a.actor_net).cuda().eval()
        learned_cp = M/'experiments/location_finding/hec_v3_study_20261115/seed101_learned/joint_actor.pt'
        learned_sha = sha(learned_cp)
        original_router.load_state_dict(torch.load(learned_cp, map_location='cuda', weights_only=True)['actor_state'])
        original_router.arm = 'learned'
        original_router.correction_enabled = True
        original_router.encoders_enabled = True
        original_router.gate_table = None
        original_router.requires_grad_(False)
        actor_before = n.digest(actor.state_dict())
        # NumPy's legacy global generator needs a uint32; default_rng retains the full seed.
        trial.engine.seed_all(cfg['seed'] % (2**32))
        rng = np.random.default_rng(cfg['seed'])
        lib = n.load_library()
        scorer_path = n.BASE/'source/frozen_evaluator.py'
        assert sha(scorer_path) == LOCK['scorer_sha256']
        scorer = load_module('frozen_uniform_diagnostic_scorer', scorer_path).contrast_scores
        contrasts = torch.tensor(rng.standard_normal((cfg['contrasts'], 4)), device='cuda')
        count = cfg['histories_per_stage']
        truth = torch.tensor(rng.standard_normal((count, 4)), device='cuda')
        noise = torch.tensor(rng.standard_normal((count, 30)), device='cuda')
        ds = torch.zeros(count, 30, 2, device='cuda')
        ys = torch.zeros(count, 30, device='cuda', dtype=torch.float64)
        prefixes = {}
        for stage in range(30):
            if stage in cfg['stages']:
                prefixes[stage] = (ds.clone(), ys.clone())
            x = lib.packed(ds, ys, stage)
            action = actor(x)
            ds[:, stage] = action
            ys[:, stage] = signal(truth.reshape(-1, 2, 2), action.double())+.5*noise[:, stage]
        rows, raw, posteriors = [], {}, []
        for stage in cfg['stages']:
            # At stage0 every prior history is identical; do not pseudoreplicate it.
            for h in range(1 if stage == 0 else count):
                trial.engine.guard(deadline)
                bd, by = [z[h].clone() for z in prefixes[stage]]
                x = lib.packed(bd[None], by[None], stage)
                theta, posterior = conditional_latents(actor, x, bd, by, stage, cfg, cfg['suffix_replicates'])
                posterior.update(stage=stage, history=h)
                posteriors.append(posterior)
                eps = torch.tensor(rng.standard_normal((len(theta), 30)), device='cuda')
                weights = [('uniform', torch.full((4,), .25, device='cuda'))]
                for k in range(4):
                    shift = torch.full((4,), -cfg['epsilon']/4, device='cuda')
                    shift[k] += cfg['epsilon']
                    weights.extend([(f'plus_E{k+1}', .25+shift), (f'minus_E{k+1}', .25-shift)])
                for k in range(4):
                    weights.append((f'only_E{k+1}', torch.eye(4, device='cuda')[k]))
                _, ri = original_router(x, collect=True)
                weights.append(('transferred_learned_router', ri['gates'][0]))
                scores, predicted, actions, shifts = {}, {}, {}, {}
                baseline_projection = None
                for label, weight in weights:
                    trial.engine.guard(deadline)
                    assert torch.isfinite(weight).all() and (weight >= 0).all()
                    torch.testing.assert_close(weight.sum(), weight.new_tensor(1.))
                    dd = bd[None].expand(len(theta), -1, -1).clone()
                    yy = by[None].expand(len(theta), -1).clone()
                    actor.diagnostic_gate = weight
                    design, info = actor(x, collect=True)
                    actions[label] = design[0].cpu().numpy()
                    projected = (weight[None, :, None]*info['projected']).sum(1)
                    if label == 'uniform':
                        baseline_projection = projected
                    shifts[label] = float((projected-baseline_projection).norm())
                    predicted[label] = float(a.critic_net(torch.cat([x, design], -1)).squeeze())
                    actor.diagnostic_gate = None
                    dd[:, stage] = design.expand(len(theta), -1)
                    yy[:, stage] = signal(theta.reshape(-1, 2, 2), dd[:, stage].double())+.5*eps[:, stage]
                    for t in range(stage+1, 30):
                        xx = lib.packed(dd, yy, t)
                        design = actor(xx)
                        assert (design.abs() <= 4.000001).all() and torch.isfinite(design).all()
                        dd[:, t] = design
                        yy[:, t] = signal(theta.reshape(-1, 2, 2), design.double())+.5*eps[:, t]
                    bench = type('Benchmark', (), {'sigma': .5, 'calls': 0})()
                    sc, _ = scorer(bench, dict(actions=list(dd.unbind(1)), observations=list(yy[..., None].unbind(1)), truth=theta), contrasts, cfg['chunk'], 'cuda')
                    scores[label] = sc[:, -1].numpy()
                    assert np.isfinite(scores[label]).all()
                    raw[f't{stage}_h{h}_{label}'] = scores[label]
                raw[f't{stage}_h{h}_theta'] = theta.cpu().numpy()
                raw[f't{stage}_h{h}_noise'] = eps.cpu().numpy()
                raw[f't{stage}_h{h}_prefix_designs'] = bd.cpu().numpy()
                raw[f't{stage}_h{h}_prefix_observations'] = by.cpu().numpy()
                labels = [w[0] for w in weights]
                half = len(theta)//2
                # Select on diagnostic discovery samples; validate on disjoint suffix samples.
                candidates = labels[:-1]  # uniform is a valid fallback; transfer assessed separately
                best = max(candidates, key=lambda k: scores[k][:half].mean())
                qbest = max(candidates, key=lambda k: predicted[k])
                validation_delta = scores[best][half:]-scores['uniform'][half:]
                qdelta = scores[qbest][half:]-scores['uniform'][half:]
                router_delta = scores['transferred_learned_router'][half:]-scores['uniform'][half:]
                measured = [float(scores[k][half:].mean()-scores['uniform'][half:].mean()) for k in candidates]
                qvalue = [predicted[k]-predicted['uniform'] for k in candidates]
                corr = spearmanr(measured, qvalue).statistic
                row = dict(stage=stage, history=h, posterior=posterior,
                    selected_on_discovery=best, critic_choice=qbest,
                    discovery_selected_validation_gain=summary(validation_delta),
                    critic_selected_validation_gain=summary(qdelta),
                    transferred_router_validation_gain=summary(router_delta),
                    critic_rank_spearman_validation=float(corr) if math.isfinite(corr) else None,
                    action_spread_max=float(max(np.linalg.norm(actions[k]-actions['uniform']) for k in labels)),
                    variants={k: dict(weight=weights[labels.index(k)][1].cpu().tolist(),
                        validation_paired_gain=summary(scores[k][half:]-scores['uniform'][half:], simultaneous=len(labels)-1),
                        critic_delta=predicted[k]-predicted['uniform'],
                        design=actions[k].tolist(), fused_projection_change=shifts[k],
                        validation_estimator_ceiling_fraction=float(np.mean(scores[k][half:] > math.log(cfg['contrasts']+1)-1e-6))) for k in labels})
                rows.append(row)
                put(out/'PROGRESS.json', dict(status='DIAGNOSING_FROZEN_CHECKPOINT', stage=stage, completed_histories=len(rows), total_histories=1+count*(len(cfg['stages'])-1), seconds=time.monotonic()-start))
                put(out/'PARTIAL_RESULTS.json', dict(rows=rows, posterior_quality=posteriors))
                print(json.dumps(dict(stage=stage, history=h, ESS=posterior.get('ESS'), best=best,
                    validation_gain=row['discovery_selected_validation_gain']['mean'], critic_choice=qbest)), flush=True)
        assert n.digest(actor.state_dict()) == actor_before
        assert sha(learned_cp) == learned_sha
        assert sha(trial.P/'runtime.pkl') == trial.config()['parent_runtime_sha256']
        assert sha(trial.P/'joint_actor.pt') == trial.config()['parent_actor_sha256']
        np.savez_compressed(out/'paired_suffix_scores.npz', **raw)
        eligible = [r for r in rows if r['posterior']['adequate'] and r['stage'] > 0]
        stage_summaries = {}
        for stage in cfg['stages']:
            chosen = [r for r in rows if r['stage'] == stage and r['posterior']['adequate']]
            if chosen:
                stage_summaries[str(stage)] = {key: summary([r[key]['mean'] for r in chosen])
                    for key in ['discovery_selected_validation_gain', 'critic_selected_validation_gain', 'transferred_router_validation_gain']}
        result = dict(status='PASS' if smoke else 'COMPLETE', protocol=cfg, seconds=time.monotonic()-start,
            rows=rows, stage_summaries=stage_summaries,
            adequate_posteriors=sum(p['adequate'] for p in posteriors), total_histories=len(rows),
            all_weights_and_paths_frozen=True, parent_checkpoints_unchanged=True,
            primary_test_bank_loaded=False, training_launched=False,
            across_histories_conditional_summary={key: summary([r[key]['mean'] for r in eligible]) for key in ['discovery_selected_validation_gain', 'critic_selected_validation_gain', 'transferred_router_validation_gain']} if eligible else {},
            limitations=['Conditional theta draws use approximate finite-pool self-normalized importance resampling;ESS thresholds are diagnostic,not a proof of posterior accuracy.',
                'Held-out suffix validation limits candidate-selection optimism;shared prior contrast-bank uncertainty and posterior approximation error are excluded from reported CIs.',
                'Transferred router comes from the separately learned101 checkpoint and different learned expert representations;transfer failure alone does not prove a router-learning bug.',
                'One-step intervention then uniform continuation;not unrestricted sequence-level routing optimization.',
                'This is a fresh diagnostic bank,not the2000-case/1M-contrast primary benchmark. Do not report these conditional EIGs as final method EIG.'])
        put(out/'RESULTS.json', result)
        text = '# Frozen uniform101 routing diagnostic\n\nNo training. Same histories, frozen expert/fusion weights, paired conditional parameter samples and future observation noise.\n\n'
        text += '| Stage | History | ESS | Discovery-best held-out gain | Critic-choice held-out gain | Transferred-router held-out gain |\n|---:|---:|---:|---:|---:|---:|\n'
        for row in rows:
            text += f"| {row['stage']} | {row['history']} | {row['posterior'].get('ESS')} | {row['discovery_selected_validation_gain']['mean']:+.5f} | {row['critic_selected_validation_gain']['mean']:+.5f} | {row['transferred_router_validation_gain']['mean']:+.5f} |\n"
        text += '\nSummary across adequate noninitial histories (descriptive,stage-stratified small sample):\n\n```json\n'+json.dumps(result['across_histories_conditional_summary'], indent=2)+'\n```\n\n'
        text += '\n'.join('- '+v for v in result['limitations'])+'\n'
        (out/'REPORT.md').write_text(text)
        print(json.dumps(dict(status=result['status'], report=str(out/'REPORT.md'), summary=result['across_histories_conditional_summary'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    run(args.smoke)
