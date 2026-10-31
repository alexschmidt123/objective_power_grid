"""Paired routing diagnostic using the unchanged common EIG evaluator.

Does not train or rerun posterior inference. Hard scores are reused only after
independently reproducing the first evaluation batch with the saved bank.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from src.research.moe_architecture_controls import ExistingRoutingView


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--max-hours', type=float, default=.5)
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    root = Path.cwd()
    c = Path(args.campaign).resolve()
    out = c/'routing_eig_diagnostics'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    paths = dict(component='experiments/10142026/10142026_moe_component_clipping/pilot',
                 slower='experiments/10142026/10142026_moe_delayed_sharpening/pilot')
    report = dict(status='running',train_seed=101,eval_seed=1001,systems=2000,L=1000000,
                  evaluator='src.research.location_evaluation.contrast_scores unchanged',
                  posterior_inference='saved hard-route secondary metrics preserved; no new inference',results={})
    for name,rel in paths.items():
        if (out/f'{name}_paired.json').exists():
            report['results'][name] = json.loads((out/f'{name}_paired.json').read_text())
            continue
        source = root/rel
        config = json.loads((source/'run_config.json').read_text())
        b = make_benchmark(config['benchmark'])
        rows = json.loads((source/'eval/moe_sboed_seed1001.json').read_text())
        assert len(rows)==2000
        bank = torch.from_numpy(np.load(source/'eval/contrast_bank_seed1001.npy'))
        assert len(bank)==1000000
        saved = torch.load(source/'models/moe_sboed.pt',map_location='cpu',weights_only=False)
        hard = ExistingRoutingView(saved,'hard')
        relaxed = ExistingRoutingView(saved,'relaxed')
        truth = torch.tensor([r['truth'] for r in rows],dtype=torch.float64)
        noise = torch.tensor([r['measurement_noise_standard'] for r in rows],dtype=torch.float64)
        hard_scores = np.array([r['spce'] for r in rows])
        st = time.perf_counter()
        r = rollout(b,hard,30,16,1001,0,truth=truth[:16],noise=noise[:16])
        assert np.allclose(np.stack([x.numpy() for x in r['actions']],axis=1),np.array([x['actions'] for x in rows[:16]]),rtol=0,atol=1e-10)
        reproduced,_ = contrast_scores(b,r,bank,16384)
        assert np.allclose(reproduced[:,-1].numpy(),hard_scores[:16],rtol=0,atol=1e-9)
        measured = time.perf_counter()-st
        estimated_remaining_seconds = measured*125*(len(paths)-len(report['results']))
        cost = dict(host='labpc',first_batch_seconds=measured,
                    estimated_remaining_seconds=estimated_remaining_seconds,
                    max_seconds=args.max_hours*3600,inference=False)
        write(out/'cost_check.json',cost)
        if estimated_remaining_seconds>args.max_hours*3600:
            raise RuntimeError('Measured routing diagnostic estimate exceeds stated cost limit')
        if args.preflight_only:
            print(json.dumps(dict(cost=cost, hard_reproduction_max_error=float(np.max(np.abs(reproduced[:,-1].numpy()-hard_scores[:16])))),indent=2),flush=True)
            return
        vals=[]
        for start in range(0,2000,16):
            r = rollout(b,relaxed,30,16,1001+start*97451,0,truth=truth[start:start+16],noise=noise[start:start+16])
            scores,_ = contrast_scores(b,r,bank,16384)
            vals.extend(scores[:,-1].tolist())
            if (start+16)%128==0 or start+16==2000:
                print(name,'relaxed',start+16,'/2000',flush=True)
        x = np.asarray(vals)
        delta = x-hard_scores
        se = float(delta.std(ddof=1)/math.sqrt(len(delta)))
        result = dict(hard_eig=float(hard_scores.mean()),relaxed_eig=float(x.mean()),
            relaxed_minus_hard=float(delta.mean()),paired_mc_se=se,
            paired_mc_95_interval=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],
            seconds=time.perf_counter()-st,relaxed_scores=vals,
            hard_reproduction_max_error=float(np.max(np.abs(reproduced[:,-1].numpy()-hard_scores[:16]))),
            checkpoint_sha256=hashlib.sha256((source/'models/moe_sboed.pt').read_bytes()).hexdigest(),
            bank_sha256=hashlib.sha256((source/'eval/contrast_bank_seed1001.npy').read_bytes()).hexdigest(),
            training_seed_uncertainty='unavailable: one training seed')
        write(out/f'{name}_paired.json',result)
        report['results'][name] = result
        write(out/'report.json',report)
    report['status']='complete'
    write(out/'report.json',report)
    print('ROUTING_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    with torch.no_grad():
        main()
