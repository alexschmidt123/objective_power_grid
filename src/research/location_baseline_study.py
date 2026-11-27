"""Reproduce the external location baselines at a prospectively fixed horizon."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.journal_checkpoints import load_checkpoint
from src.research.journal_run import main as journal_train
from src.research.location_evaluation import contrast_scores
from src.results.run_directory import allocate_run

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('fixed', 'dad', 'rl_sboed', 'vsoed', 'scorebed', 'random')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def locked_draws(path, protocol):
    if sha(path) != protocol['parent_sha256']:
        raise ValueError('Original evaluation bank digest mismatch')
    if sha(ROOT/'src/research/location_evaluation.py') != protocol['scorer_sha256']:
        raise ValueError('Canonical scorer digest mismatch')
    with np.load(path, allow_pickle=False) as bank:
        truth, noise, contrasts = (bank[k] for k in ('truth', 'noise', 'contrasts'))
    if (truth.shape, noise.shape, contrasts.shape) != ((2000, 4), (2000, 30), (1000000, 4)):
        raise ValueError('Unexpected canonical location bank layout')
    arrays = [np.ascontiguousarray(a) for a in (truth, noise[:, :protocol['noise_prefix'], None], contrasts)]
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(a.tobytes())
    if digest.hexdigest() != protocol['view_sha256']:
        raise ValueError('Locked T15 truth, noise or contrasts changed')
    return arrays, digest.hexdigest()


def train(args, cfg):
    root = args.experiments.resolve()
    proto = cfg['research']['journal_protocol']
    if args.method == 'random':
        out = allocate_run(root, args.config.stem, args.horizon, [args.method])
        (out/'models').mkdir()
        save(out/'run_config.json', dict(benchmark=cfg, train_seed=args.seed, methods=[args.method],
                                         horizon=args.horizon, smoke=False))
        return out
    options = SimpleNamespace(journal_phase='development', seed=args.seed, eval_seeds=None, T=args.horizon,
                              variant=None, methods=args.method, smoke=False, output=None,
                              config=args.config, evaluation_cases=None, evaluation_bank=None, train_only=True)
    return journal_train(options, cfg, ROOT, experiment_root=root)


def native_checkpoint(out, args, cfg):
    meta = json.loads((out/'run_config.json').read_text())
    if (meta['methods'], meta['train_seed'], meta['horizon'], meta['smoke']) != ([args.method], args.seed, args.horizon, False):
        raise ValueError('Training receipt does not identify this cell')
    if args.method == 'random':
        return None, {'training_status':'untrained_random_policy', 'training_seconds':0}, None
    cp = out/'models'/f'{args.method}.pt'
    rec = json.loads(cp.with_name(f'{args.method}_training.json').read_text())
    if sha(cp) != rec['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since training')
    if not rec['native_schedule_completed']:
        raise RuntimeError('Native schedule incomplete; primary comparison ineligible')
    if args.method == 'scorebed' and (rec['actual_score_updates'], rec['actual_policy_updates'], rec['completed_restarts']) != (50000, 100000, 5):
        raise ValueError('Incomplete ScoreBED restarts')
    policy = load_checkpoint(cp, make_benchmark(cfg)).eval()
    # Upstream vsOED stores bounds as ordinary tensor attributes.
    buffer = io.BytesIO()
    torch.save(policy, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location='cuda', weights_only=False).eval(), rec, sha(cp)


@torch.inference_mode()
def evaluate(args, out, cfg, arrays, bank_hash):
    p, record, checkpoint_hash = native_checkpoint(out, args, cfg)
    proto = cfg['research']['journal_protocol']['final_evaluation']
    theta, noise, bank = [torch.as_tensor(a, device='cuda', dtype=torch.float64) for a in arrays]
    h, batch = args.horizon, proto['batch']
    rng_seed = proto['policy_rng_base'] + (args.seed + 10000 if args.method=='random' else 0)
    online = scoring = 0.
    scores = []
    with torch.device('cuda'):
        bench = make_benchmark(cfg)
        def episode(start, n):
            return rollout(bench,p,h,n,rng_seed+start*97451,0,truth=theta[start:start+n],
                           noise=noise[start:start+n],method=args.method)
        episode(0,1)
        torch.cuda.synchronize()
        tick = time.monotonic()
        for i in range(16):
            episode(i,1)
        torch.cuda.synchronize()
        single_latency = (time.monotonic()-tick)/16
        for i in range(0,len(theta),batch):
            tick = time.monotonic()
            result = episode(i,min(batch,len(theta)-i))
            torch.cuda.synchronize()
            online += time.monotonic()-tick
            tick = time.monotonic()
            curve,_ = contrast_scores(bench,result,bank,proto['contrast_chunk'],'cuda')
            torch.cuda.synchronize()
            scoring += time.monotonic()-tick
            scores.append(curve[:,-1].numpy())
    scores = np.concatenate(scores)
    if scores.shape != (proto['systems'],) or not np.isfinite(scores).all() or scores.max() > math.log1p(len(bank))+1e-9:
        raise ValueError('Invalid terminal full-horizon EIG')
    dest = out/'eval'
    dest.mkdir(exist_ok=True)
    np.save(dest/'scores.npy',scores)
    save(dest/'primary.json',dict(method=args.method,seed=args.seed,horizon=h,eig_nats=float(scores.mean()),
         case_mcse=float(scores.std(ddof=1)/math.sqrt(len(scores))),protocol=proto,bank_view_sha256=bank_hash,
         policy_rng_base=rng_seed,checkpoint_sha256=checkpoint_hash,score_vector_sha256=sha(dest/'scores.npy'),
         training_status=record.get('training_status'),primary_episodes=record.get('generated_training_trajectories',0),
         offline_seconds=record.get('total_offline_seconds',record.get('training_seconds')),
         charged_training_seconds=record.get('compute_budget',{}).get('charged_seconds'),
         gpu=torch.cuda.get_device_name(),batch=batch,online_rollout_seconds=online,
         single_episode_seconds=single_latency,batched_episodes_per_second=len(theta)/online,
         timing_scope='GPU-synchronized full policy and simulator rollout; batch-one mean of 16 after warmup. Isolated action latency unmeasured.',
         scorer_seconds=scoring,process_cpu_seconds=time.process_time(),
         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         torch_process_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
         revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         job=os.getenv('SLURM_JOB_ID'),config_sha256=sha(args.config)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--location-baselines',action='store_true',required=True)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--method',choices=METHODS,required=True)
    ap.add_argument('--seed',type=int,choices=range(101,106),required=True)
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--experiments',type=Path,default=ROOT/'experiments')
    ap.add_argument('--run-dir',type=Path,help='Score an already-trained cell')
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    args.horizon=cfg['experiment']['step_number']
    proto=cfg['research']['journal_protocol']
    if args.horizon!=15 or args.horizon!=proto['final_evaluation']['noise_prefix'] or proto['native_vsoed_profile']['n_incre']!=args.horizon:
        raise ValueError('Inconsistent location horizon')
    arrays, digest=locked_draws(args.bank,proto['final_evaluation'])
    if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.receipt.exists():
        previous=json.loads(args.receipt.read_text())
        if (previous['method'],previous['seed'],previous['config_sha256']) != (args.method,args.seed,sha(args.config)):
            raise ValueError('Existing receipt identifies a different cell')
        if args.run_dir and args.run_dir.resolve()!=Path(previous['directory']).resolve():
            raise ValueError('Existing receipt identifies another training run')
        args.run_dir=Path(previous['directory'])
        if (args.run_dir/'eval/primary.json').exists():
            raise FileExistsError('Evaluation already complete')
    out=args.run_dir or train(args,cfg)
    save(args.receipt,dict(directory=str(out),method=args.method,seed=args.seed,config_sha256=sha(args.config)))
    evaluate(args,out,cfg,arrays,digest)


if __name__=='__main__':
    main()
