"""Maintained dispatch for the active non-reset grid protocol and SIR benchmark."""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from src.config import load_config, resolve_config_path

ROOT=Path(__file__).resolve().parents[1]


def main():
    argv=sys.argv[1:]
    from tools.storage_policy import enforce_storage_policy
    argv = enforce_storage_policy(ROOT, argv)
    sys.argv = [sys.argv[0], *argv]
    for arg in argv:
        flag=arg.split('=',1)[0].replace('_','-')
        if flag.startswith(('--reuse','--resume','--checkpoint','--load-checkpoint','--skip-training')):
            raise SystemExit('run.sh/sweep_run.sh require fresh experiments; checkpoint/result reuse and resume are forbidden.')
    if '--sweep' not in argv:
        p=argparse.ArgumentParser(add_help=False)
        p.add_argument('--config',default='configs/ieee9_research_eig.yaml')
        known,_=p.parse_known_args(argv)
        cfg=load_config(resolve_config_path(known.config))
        if cfg.raw.get('research',{}).get('enabled',False) and (not any(a.split('=')[0]=='--evaluate-from' for a in argv) or cfg.raw.get('system',{}).get('name')=='location_finding'):
            from src.research.runner import main as research
            research(argv if any(a.split("=")[0]=="--config" for a in argv) else ["--config",known.config,*argv])
            return
        if str(cfg.raw.get('system',{}).get('name','')).lower().startswith('sir'):
            raise SystemExit(subprocess.call(['bash',str(ROOT/'scripts/sir_run.sh'),*argv],cwd=ROOT))
        from src.objectives.eig.continuous_eig import main as online
        online()
        return
    if any(a.split('=')[0]=='--evaluate-from' for a in argv):
        raise SystemExit('--evaluate-from is per-run; use run.sh with one matching source run')
    argv.remove('--sweep')
    p=argparse.ArgumentParser(description='Explicit Cartesian sweep; each cell starts a fresh run.sh experiment with its own training and evaluations.')
    p.add_argument('--configs','--config',default='ieee9_research_eig')
    p.add_argument('--T',default='3')
    p.add_argument('--N_obs','--N-obs',default='0')
    p.add_argument('--noise_sigma','--noise-sigma',default=None)
    p.add_argument('--seed',default='101')
    p.add_argument('--eval-seeds',default='1001')
    p.add_argument('--coverage',default=None)
    p.add_argument('--campaign-root',help='Date-grouped parent for sweep cells; never reused')
    args,other=p.parse_known_args(argv)
    if '--output' in other:p.error('Each sweep cell allocates its own output; do not pass --output')
    axes=[args.configs.split(','),args.T.split(','),args.N_obs.split(','),args.noise_sigma.split(',') if args.noise_sigma else [None],args.seed.split(','),args.coverage.split(',') if args.coverage else [None]]
    for config,T,nobs,sigma,seed,coverage in itertools.product(*axes):
        cmd=['bash',str(ROOT/'run.sh'),'--config',config,'--T',T,'--N_obs',nobs,
             '--seed',seed,'--eval-seeds',args.eval_seeds,*other]
        if args.campaign_root:
            from pathlib import Path
            parent=Path(args.campaign_root)
            cell=parent/(Path(config).stem+f'_T{T}_train{seed}_Nobs{nobs}'+(f'_sigma{sigma}' if sigma else '')+(f'_coverage{coverage}' if coverage else ''))
            cmd+=['--output',str(cell)]
        if sigma is not None:cmd+=['--noise_sigma',sigma]
        if coverage is not None:cmd+=['--coverage',coverage]
        subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':main()
