"""Compare completed location development policies on identical evaluation cases."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

def read(p):return json.loads(p.read_text())
def checksum(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def compare(candidate,dad,vsoed,seed):
    sources={'moe_sboed':candidate,'matched_dense':candidate,'uniform_moe':candidate,'dad':dad,'vsoed':vsoed}
    rows={};summaries={};configs={};banks={}
    for method,p in sources.items():
        if (p/'exit_code').read_text().strip()!='0':raise ValueError(f'Incomplete source: {p}')
        configs[method]=read(p/'run_config.json')
        rows[method]=read(p/'eval'/f'{method}_seed{seed}.json')
        summaries[method]=next(r for r in read(p/'summary.json') if r['method']==method and r['eval_seed']==seed)
        banks[method]=checksum(p/'eval'/f'contrast_bank_seed{seed}.npy')
    reference=rows['moe_sboed'];comparisons={}
    for method,cfg in configs.items():
        for key in ['train_seed','horizon']:
            if cfg[key]!=configs['moe_sboed'][key]:raise ValueError(f'Different {key}: {method}')
        if cfg['smoke']:raise ValueError('Smoke is not performance evidence')
        for key in ['system','N','observation','location_finding']:
            if cfg['benchmark'][key]!=configs['moe_sboed']['benchmark'][key]:raise ValueError(f'Different physical model {key}: {method}')
    for method,cases in rows.items():
        if len(cases)!=len(reference):raise ValueError('Different case counts')
        for a,b in zip(reference,cases):
            for key in ['trajectory_id','truth','measurement_noise_standard','eval_seed']:
                if a[key]!=b[key]:raise ValueError(f'Unpaired {method}: {key}')
            if len(a['actions'])!=len(b['actions']):raise ValueError('Different horizons')
        if summaries[method]['contrasts']!=summaries['moe_sboed']['contrasts']:raise ValueError('Different EIG budgets')
        if banks[method]!=banks['moe_sboed']:raise ValueError('Different contrast banks')
        if method=='moe_sboed':continue
        delta=[a['spce']-b['spce'] for a,b in zip(reference,cases)]
        if not all(math.isfinite(x) for x in delta):raise ValueError('Nonfinite EIG')
        mean=statistics.mean(delta);se=statistics.stdev(delta)/math.sqrt(len(delta))
        comparisons[method]=dict(mean_difference=mean,paired_mc_standard_error=se,
                                  approximate_mc_95_interval=[mean-1.96*se,mean+1.96*se])
    return dict(evaluation_seed=seed,cases=len(reference),contrasts=summaries['moe_sboed']['contrasts'],
        interpretation='Development evaluation used for method selection; single training seed; intervals measure case variation, not training-seed uncertainty',
        same_contrast_bank=True,contrast_bank_sha256=banks['moe_sboed'],
        source_runs={k:str(v) for k,v in sources.items()},summaries=summaries,comparisons=comparisons)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--dad',type=Path,required=True);p.add_argument('--vsoed',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(a.candidate,a.dad,a.vsoed,a.seed)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for method,r in result['summaries'].items():print(f'{method}: EIG={r["mean_spce"]:.6f} nats')
    for method,r in result['comparisons'].items():print(f'MoE minus {method}: {r["mean_difference"]:.6f}, paired MC95 {r["approximate_mc_95_interval"]}')
