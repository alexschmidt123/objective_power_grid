"""Screen configured MOCU decisions using existing banks, without training."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from datetime import datetime
import json
from src.config import load_config_for_run
from src.objectives.mocu.context import build_context_from_config
from src.objectives.mocu.preflight import enforce_decision_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ieee9_mocu.yaml')
    parser.add_argument('--bank-root', type=Path, default=ROOT)
    parser.add_argument('--T', type=int, default=3)
    args = parser.parse_args()
    cfg = load_config_for_run(args.config, ROOT, step_number=args.T)
    for key in ('dataset_dir', 'reuse_bank_dir', 'mocu_dataset_dir'):
        path = (cfg.raw.get('data') or {}).get(key)
        if path:
            cfg.raw['data'][key] = str((args.bank_root / path).resolve())
    observation = cfg.raw.setdefault('observation', {})
    observation.update(observation.get('objective_based', {}))
    # Skip Fixed calibration only; retain production-size banks and quality gates.
    cfg.raw.setdefault('experiment', {})['allow_trivial_fixed'] = True
    cfg.raw['experiment']['experiment_type'] = 'objective_based'
    sigma = str(observation.get('noise_sigma', .005)).replace('.', 'p')
    stamp = datetime.now().strftime('%m%d%Y_%H%M%S')
    out = ROOT/'experiments'/f'{stamp}_{cfg.name}_preflight_Uctrl_T{args.T}_Nobs{observation.get("N_obs",5)}_sigma{sigma}'
    ctx = build_context_from_config(cfg, project_root=ROOT, out_dir=out,
                                    smoke=False, experiment_type='objective_based')
    report = enforce_decision_preflight(ctx)
    print(json.dumps(report, indent=2))
    print('REPORT=' + str(out/'diagnostics'/'mocu_preflight.json'))


if __name__ == '__main__':
    main()
