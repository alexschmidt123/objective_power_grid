"""Integration smoke test on existing banks; writes only a new test run."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from datetime import datetime
from src.config import load_config_for_run
from src.objectives.mocu.context import build_context_from_config
from src.objectives.mocu.train import train_policy
from src.objectives.mocu.evaluate import run_full_evaluation
from src.layout import write_run_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-root', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config_for_run('configs/ieee9_mocu.yaml', ROOT, step_number=3)
    for key in ('dataset_dir', 'reuse_bank_dir', 'mocu_dataset_dir'):
        cfg.raw['data'][key] = str((args.bank_root / cfg.raw['data'][key]).resolve())
    cfg.raw['observation'].update(cfg.raw['observation']['objective_based'])
    cfg.raw['experiment']['allow_trivial_fixed'] = True
    cfg.raw['training']['device'] = 'cpu'
    stamp = datetime.now().strftime('%m%d%Y_%H%M%S')
    out = ROOT/'experiments'/f'{stamp}_correctness_smoke_Uctrl_T3_Nobs5_sigma0p005'
    ctx = build_context_from_config(cfg, project_root=ROOT, out_dir=out,
                                    smoke=False, experiment_type='objective_based')
    write_run_config(out, cfg, Path(cfg.raw['data']['dataset_dir']),
                     experiment_type='objective_based', extra={'smoke': True, 'seed': 101})
    for method in ('DAD', 'RL-sBOED'):
        train_policy(ctx, method=method, seed=101, smoke=True)
    ctx.test_systems = ctx.test_systems[:4]  # Bound the smoke oracle as well as rollouts.
    report = run_full_evaluation(ctx, methods=['dad','rl_sboed','step_dad','myopic','fixed','random'],
                                 smoke=True, skip_cuda_safety=True, eval_seed=1001)
    assert report['metric_schema'] == 'realized_operational_regret_v2'
    print('INTEGRATION_SMOKE_OK', out)


if __name__ == '__main__':
    main()
