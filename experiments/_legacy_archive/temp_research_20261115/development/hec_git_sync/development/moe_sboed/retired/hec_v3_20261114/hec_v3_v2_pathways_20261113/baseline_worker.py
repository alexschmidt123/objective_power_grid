"""Historical baseline namespace isolated from native V2 package imports."""
import argparse,hashlib,importlib.util,io,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=pathlib.Path.home()/'Documents/objective_power_grid_temp';base=root/'scratch_runs/moe_hec_v2_icml_20261109'
    inventory=json.loads((base/'inputs/HISTORICAL_BASELINES.json').read_text());row=next(r for r in inventory['rows'] if r['method']==args.method)
    assert sha(row['checkpoint'])==row['checkpoint_sha256']
    if row.get('arrays_path'):assert sha(row['arrays_path'])==row['arrays_sha256']
    sys.path.insert(0,inventory['source_root'])
    from src.research.journal_checkpoints import load_checkpoint
    from src.research.benchmarks import make_benchmark
    from src.research.engine import rollout
    spec=importlib.util.spec_from_file_location('frozen_LF_score',base/'source/frozen_evaluator.py');score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    benchmark=make_benchmark(inventory['physics'])
    policy=torch.load(row['checkpoint'],map_location='cpu',weights_only=False) if row.get('loader')=='object' or args.method=='rl_sboed' else load_checkpoint(row['checkpoint'],benchmark)
    buffer=io.BytesIO();torch.save(policy,buffer);buffer.seek(0);policy=torch.load(buffer,map_location='cuda',weights_only=False).eval()
    if hasattr(policy,'stochastic'):policy.stochastic=False
    cfg=json.loads((args.output/'CONFIG.json').read_text());cases=np.load(args.output/'EVALUATION_CASES.npz')
    count=2 if args.smoke else len(cases['truth']);contrasts_count=31 if args.smoke else cfg['contrasts']
    array=np.random.default_rng(cfg['contrast_seed']).standard_normal((contrasts_count,4));contrasts=torch.tensor(array,device='cuda');del array
    values=[];start=time.monotonic()
    with torch.no_grad(),torch.device('cuda'):
        for first in range(0,count,16):
            truth=torch.tensor(cases['truth'][first:min(first+16,count)],device='cuda',dtype=torch.float64)
            noise=torch.tensor(cases['noise'][first:min(first+16,count),:,None],device='cuda',dtype=torch.float64)
            result=rollout(benchmark,policy,30,len(truth),202611134299+first*97451,0,truth=truth,noise=noise,method='rl_sboed' if args.method=='rl_sboed' else None)
            scores,_=score.contrast_scores(benchmark,result,contrasts,16384,'cuda');values.extend(scores[:,-1].numpy())
    assert sha(row['checkpoint'])==row['checkpoint_sha256']
    np.save(args.output/f'{args.method}_scores.npy',np.array(values))
    receipt=dict(status='COMPLETE',method=args.method,cases=count,contrasts=contrasts_count,mean=float(np.mean(values)),
                 checkpoint_sha256=row['checkpoint_sha256'],historical_not_compute_matched=True,seconds=time.monotonic()-start)
    (args.output/f'{args.method}_BASELINE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
