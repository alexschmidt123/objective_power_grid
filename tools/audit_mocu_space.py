#!/usr/bin/env python3
"""Audit adaptive/non-myopic finite-loss MOCU room before neural training."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import torch
from src.config import load_config_for_run
from src.layout import make_experiment_dir_name, write_run_config
from src.objectives.mocu.context import build_context_from_config
from tools.audits.space import AuditBudget, SpacePlanner, summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ieee9_mocu.yaml')
    parser.add_argument('--horizons', default='2,3')
    parser.add_argument('--seeds', default='101,202,303')
    parser.add_argument('--phase', choices=['screen', 'confirm'], default='screen')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--inner', type=int)
    parser.add_argument('--outer', type=int)
    parser.add_argument('--first-candidates', type=int)
    parser.add_argument('--eval-systems', type=int)
    parser.add_argument('--fixed-restarts', type=int)
    parser.add_argument('--calibration', type=int)
    parser.add_argument('--practical-fraction', type=float, default=.05)
    args = parser.parse_args()
    horizons = sorted(set(map(int, args.horizons.split(','))))
    seeds = list(map(int, args.seeds.split(',')))
    if not seeds or min(horizons)<2:
        parser.error('Use at least one seed and horizons >=2')
    cfg = load_config_for_run(args.config, ROOT, step_number=max(horizons))
    observation = cfg.raw.setdefault('observation', {})
    observation.update(observation.get('objective_based', {}))
    control = cfg.raw.get('control') or {}
    training = cfg.training_for('objective_based')
    if (control.get('robust_rule') != 'quantile' or control.get('alpha') != .05
        or training.get('undercontrol_penalty') != 20 or training.get('violation_penalty') != 0
        or control.get('safety_margin', 0) != 0 or not control.get('snap_up', True)):
        parser.error('This audit requires the aligned finite-loss 95% quantile protocol')
    # Our separate GPU Fixed calibration replaces the production CPU search.
    cfg.raw['experiment']['allow_trivial_fixed'] = True
    name = make_experiment_dir_name(cfg.name+'_space_audit_'+args.phase,
        'objective_based', max(horizons), n_obs=int(observation['N_obs']),
        noise_sigma=float(observation['noise_sigma']))
    out = ROOT/'experiments'/name
    ctx = build_context_from_config(cfg, project_root=ROOT, out_dir=out,
                                    smoke=False, experiment_type='objective_based')
    if max(horizons)>ctx.n_actions:
        parser.error('Horizon cannot exceed the non-repeated action catalog')
    budget = (AuditBudget(inner=8, outer=4, first_candidates=6, calibration=128,
                          histories_per_batch=32) if args.phase=='screen' else
              AuditBudget(inner=32, outer=16, first_candidates=12, calibration=256,
                          histories_per_batch=16))
    for field in ('inner', 'outer', 'first_candidates', 'fixed_restarts', 'calibration'):
        value = getattr(args, field)
        if value is not None: setattr(budget, field, value)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    planner = SpacePlanner(ctx.centres_support, ctx.U_support, ctx.u_grid,
        sigma=ctx.sigma_y, alpha=ctx.alpha, penalty=ctx.undercontrol_penalty,
        device=device, budget=budget)
    nval = len(ctx.validation_systems)
    half = nval//2
    indices = np.arange(half) if args.phase=='screen' else np.arange(half, nval)
    if args.eval_systems is not None and args.eval_systems < 4:
        parser.error('--eval-systems must be at least four')
    if args.eval_systems:
        indices = indices[:args.eval_systems]
    if len(indices)<4:
        parser.error('At least four independent validation systems are required')
    systems = [ctx.validation_systems[i] for i in indices]
    centres = np.stack([s['obs_clean'] for s in systems])
    required = np.array([s['u_req'] for s in systems])
    prior_risk, prior_control = planner.risk(planner.prior())
    prior_risk = float(prior_risk[0])
    doc = {'schema': 'finite_loss_adaptive_space_v2', 'status': 'running',
        'phase': args.phase, 'system': ctx.system, 'device': device,
        'budget': asdict(budget), 'horizons': horizons, 'audit_noise_seeds': seeds,
        'n_support': planner.P, 'n_actions': planner.A, 'N_obs': ctx.n_obs,
        'noise_sigma': ctx.sigma_y, 'alpha': ctx.alpha, 'penalty': ctx.undercontrol_penalty,
        'terminal_rule': 'quantile',
        'metric': 'candidate_grid_bank_regret',
        'loss_formula': 'u + 20 * max(U_grid - u, 0) - U_grid',
        'oracle_scope': 'validation control-bank U_grid, not continuous bisection oracle',
        'second_stage_value': 'independent fantasies after action selection', 'validation_indices': indices.tolist(),
        'prior_bayes_risk': prior_risk, 'prior_control': float(prior_control[0]),
        'practical_gain_reference': max(1e-4, args.practical_fraction*prior_risk),
        'support_sha256': hashlib.sha256(np.ascontiguousarray(ctx.U_support).tobytes()+
            np.ascontiguousarray(ctx.centres_support).tobytes()).hexdigest(),
        'validation_sha256': hashlib.sha256(centres.tobytes()+required.tobytes()).hexdigest(),
        'planning': 'receding two-step; immediate-best plus diverse first candidates; all second actions',
        'fixed': 'greedy multistart on independent in-support calibration simulations',
        'evaluation': 'off-support validation half; final test bank untouched; paired action/step noise',
        'claim_scope': 'approximate planner witness, not an upper bound on DAD-family performance',
        'physical_safety_certified': False, 'structure': planner.structure_summary(), 'results': {}}
    write_run_config(out, cfg, Path(ctx.cfg.raw['data']['dataset_dir']),
        experiment_type='objective_based', extra={'audit_protocol': doc, 'methods': []})
    started = time.perf_counter()
    for horizon in horizons:
        print(f'[audit] calibrating Fixed for T={horizon}', flush=True)
        fixed, calibration_loss = planner.fixed_sequence(horizon, seed=104729+horizon)
        runs=[]
        for seed in seeds:
            run=planner.evaluate(centres, required, horizon, fixed, seed)
            arrays = {f'{method}_{key}': value for method,row in run.items()
                      for key,value in row.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(out/f'paired_T{horizon}_seed{seed}.npz', **arrays)
            runs.append(run)
        summary=summarize_runs(runs, prior_risk, budget)
        summary.update(fixed_sequence=fixed, fixed_calibration_loss=calibration_loss)
        threshold=doc['practical_gain_reference']
        gaps=[summary['adaptive_gain'], summary['nonmyopic_gain']]
        if all(g['ci95'][0]>0 and g['mean']>=threshold for g in gaps):
            verdict='positive_planner_signal_needs_confirmation' if args.phase=='screen' else 'positive_planner_signal'
        elif any(g['ci95'][1]<threshold for g in gaps):
            verdict='material_gain_not_shown_by_this_planner'
        else:
            verdict='inconclusive_increase_planning_or_validation_budget'
        summary['verdict']=verdict
        doc['results'][str(horizon)]=summary
        (out/'audit_summary.json').write_text(json.dumps(doc, indent=2)+'\n')
        print(f'[audit] T={horizon}: {verdict}', flush=True)
    doc.update(status='complete', elapsed_seconds=time.perf_counter()-started)
    (out/'audit_summary.json').write_text(json.dumps(doc, indent=2)+'\n')
    lines=['# Finite-loss MOCU space audit', '',
        f"Phase: {args.phase}. {len(indices)} validation systems × {len(seeds)} observation-noise seeds; no policy training.",
        f'Posterior support: {planner.P}; actions: {planner.A}; prior Bayes risk: {prior_risk:.6f}.', '',
        'Positive gains favor the adaptive policy in each comparison. Intervals cluster repeated seeds by validation system.',
        'Metric: candidate-grid bank regret u + 20 max(U_grid-u, 0) - U_grid.',
        'This uses bank U_grid, not the continuous bisection oracle used in final realized operational regret.',
        'This is a diagnostic, not a final performance table or physical-safety certification.', '']
    for horizon, summary in doc['results'].items():
        lines += [f'## Horizon {horizon}', '']
        for label,key in [('Myopic adaptation over calibrated Fixed','myopic_adaptive_gain'),
                          ('Combined lookahead gain over calibrated Fixed','adaptive_gain'),
                          ('Non-myopic gain over Myopic','nonmyopic_gain')]:
            g=summary[key]
            lines.append(f"- {label}: {g['mean']:.6f}; paired 95% interval [{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}].")
        lines += [f"- Status: `{summary['verdict']}`.",
                  f"- Lookahead bank-based coverage: {summary['methods']['lookahead']['bank_coverage']:.3f}.",
                  f"- Lookahead median posterior ESS: {summary['methods']['lookahead']['median_posterior_ess']:.2f}.", '']
    lines += ['## Limits', '', 'A weak approximate planner cannot prove there is no adaptive or non-myopic space.',
        'Screen and confirm use disjoint validation halves. Freeze settings before confirmation; final held-out tests remain separate.',
        'A positive planning signal does not guarantee DAD can learn it. More distinct action sequences alone do not establish valuable adaptivity.',
        'Fixed is approximate; a stronger Fixed calibration may reduce the apparent adaptive advantage.',
        'Intervals are unadjusted per comparison and conditional on the banks, planning seeds and Fixed calibration.',
        'Confirmation halves are consumed after inspection; further tuning requires fresh validation for confirmatory claims.',
        'The prior Bayes risk is an in-model perfect-information ceiling, not a bound on off-support empirical gains.']
    (out/'audit_report.md').write_text('\n'.join(lines)+'\n')
    print('AUDIT_REPORT='+str(out/'audit_report.md'), flush=True)


if __name__=='__main__':
    main()
