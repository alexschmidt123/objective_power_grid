"""Frozen hard/relaxed interventions using the unchanged common EIG function.

Reads the complete paired cases and contrast bank from a completed root run.
No training, posterior inference, contrast subsampling or estimator modification.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from src.research.moe_family import load_policy
from tools.report_moe_family import summarize_rows

def paired(a,b):
    d=np.asarray(a)-np.asarray(b);se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else None
    return dict(mean=float(d.mean()),standard_error=se,conditional_95_interval=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)] if se is not None else None)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--variant',required=True)
    ap.add_argument('--seed',type=int,default=1001);ap.add_argument('--output',required=True)
    ap.add_argument('--mode',default='learned_switch',choices=['learned_switch','always_reroute','always_continue'])
    ap.add_argument('--relaxed',action='store_true');ap.add_argument('--compare',action='append',default=[])
    args=ap.parse_args();torch.set_num_threads(1)
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    assert (run/'exit_code').read_text().strip()=='0','Root evaluation must complete first'
    cfg=json.loads((run/'run_config.json').read_text());b=make_benchmark(cfg['benchmark'])
    if b.name!='location_finding':raise ValueError('Frozen route evaluator currently supports location only')
    checkpoint=run/'models'/(args.variant+'.pt');p=load_policy(checkpoint,b,mode=args.mode,relaxed=args.relaxed)
    native=json.loads((run/'eval'/f'{args.variant}_seed{args.seed}.json').read_text())
    bank_path=run/'eval'/f'contrast_bank_seed{args.seed}.npy';bank=torch.from_numpy(np.load(bank_path))
    truth=torch.tensor([r['truth'] for r in native],dtype=torch.float64)
    noise=torch.tensor([r['measurement_noise_standard'] for r in native],dtype=torch.float64)
    rows=[];started=time.perf_counter();settings=cfg['settings'];h=cfg['horizon']
    for start in range(0,len(native),settings['evaluation_batch_size']):
        n=min(settings['evaluation_batch_size'],len(native)-start);routes=[]
        def hook(module,inputs,output):
            r=module.last_route
            if r is not None:routes.append(dict(stage=inputs[1],weights=r['weights'].tolist(),selected_expert=r['weights'].argmax(-1).tolist(),strategy={k:v.tolist() for k,v in r['strategy'].items()}))
        handle=p.register_forward_hook(hook)
        try:r=rollout(b,p,h,n,args.seed+start*97451,0,truth=truth[start:start+n],noise=noise[start:start+n])
        finally:handle.remove()
        info,_=contrast_scores(b,r,bank,settings['evaluation_contrast_chunk'])
        for i in range(n):rows.append(dict(trajectory_id=start+i,spce=float(info[i,-1]),
            routing=[dict(stage=z['stage'],weights=z['weights'][i],selected_expert=z['selected_expert'][i],strategy={k:v[i] for k,v in z['strategy'].items()}) for z in routes]))
        if start==0 or start%256==0:print(args.variant,args.mode,args.relaxed,start+n,flush=True)
    scores=[r['spce'] for r in rows];se=float(np.std(scores,ddof=1)/math.sqrt(len(scores))) if len(scores)>1 else None
    summary=dict(canonical_variant=p.variant,mode=args.mode,relaxed=args.relaxed,systems=len(scores),contrasts=len(bank),mean_eig=float(np.mean(scores)),mc_standard_error=se,
        difference_from_native_hard=paired(scores,[r['spce'] for r in native]),seconds=time.perf_counter()-started,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),contrast_bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        case_identity_sha256=hashlib.sha256(json.dumps([[r['truth'],r['measurement_noise_standard']] for r in native],separators=(',',':')).encode()).hexdigest(),comparisons={},
        limitations=['Frozen intervention may induce out-of-distribution histories','MC uncertainty is not training-seed variability','Evaluation seed1001 is development data'])
    for path in args.compare:
        other=json.loads(Path(path).read_text())
        if len(other)!=len(native) or any(a['truth']!=c['truth'] or a['measurement_noise_standard']!=c['measurement_noise_standard'] for a,c in zip(native,other)):
            raise ValueError('Comparison trajectories are not paired')
        other_bank=Path(path).parent/f'contrast_bank_seed{args.seed}.npy'
        if hashlib.sha256(other_bank.read_bytes()).hexdigest()!=summary['contrast_bank_sha256']:raise ValueError('Comparison contrast bank differs')
        summary['comparisons'][str(path)]=paired(scores,[r['spce'] for r in other])
    (out/'scores.json').write_text(json.dumps(rows,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'management.json').write_text(json.dumps(summarize_rows(rows),indent=2,allow_nan=False)+'\n')
    (out/'exit_code').write_text('0\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
