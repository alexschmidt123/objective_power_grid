"""Cheap decision-sensitivity screen before spending a policy-training budget."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def decision_preflight(ctx, *, rollouts=64, seed=104729):
    from src.objectives.mocu.context import terminal_u_ctrl, update_posterior_vector
    rng = np.random.default_rng(seed)
    actions, losses = [], []
    # Use only validation systems, never the final held-out test set.
    systems = ctx.validation_systems
    for i in range(rollouts):
        system = systems[i % len(systems)]
        log_w = ctx.log_p0.copy()
        for action in rng.choice(ctx.n_actions, ctx.horizon, replace=False):
            observation = np.asarray(system['obs_clean'][action]) + rng.normal(
                0, ctx.sigma_y, size=ctx.obs_dim)
            log_w = update_posterior_vector(ctx, log_w, int(action), observation)
        u = float(terminal_u_ctrl(ctx, log_w))
        required = float(system['u_req'])
        shortfall = max(required-u, 0.)
        loss = u + ctx.undercontrol_penalty*shortfall + ctx.violation_penalty*(shortfall > 0)-required
        actions.append(u); losses.append(loss)
    finite = bool(np.isfinite(actions).all() and np.isfinite(losses).all())
    report = {'schema': 'mocu_decision_preflight_v1', 'robust_rule': ctx.robust_rule,
              'alpha': ctx.alpha, 'rollouts': rollouts, 'seed': seed,
              'source': 'validation_random_designs', 'finite': finite,
              'n_unique_controls': int(len(np.unique(actions))),
              'control_min': float(min(actions)), 'control_max': float(max(actions)),
              'mean_realized_regret': float(np.mean(losses)),
              'decision_degenerate': bool(np.ptp(actions) <= 1e-12),
              'physical_safety_certified': False}
    folder = Path(ctx.out_dir)/'diagnostics'; folder.mkdir(exist_ok=True, parents=True)
    (folder/'mocu_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def enforce_decision_preflight(ctx):
    settings = ctx.cfg.training_for(getattr(ctx, "experiment_type", "objective_based"))
    report = decision_preflight(ctx)
    if not report['finite']:
        raise RuntimeError('MOCU preflight produced non-finite controls or losses')
    if report['decision_degenerate']:
        message = ('MOCU preflight found constant terminal controls on validation random designs. '
                   'This is a diagnostic screen, not proof that no design can help. '
                   'Review diagnostics/mocu_preflight.json before a full training run.')
        if settings.get('require_decision_sensitivity', True):
            raise RuntimeError(message + ' Set training.objective_based.require_decision_sensitivity=false '
                               'only for an intentional degenerate-control study.')
        print('[mocu-preflight] WARNING: ' + message)
    else:
        print(f"[mocu-preflight] {report['n_unique_controls']} controls, "
              f"range [{report['control_min']:.6f}, {report['control_max']:.6f}]")
    return report
