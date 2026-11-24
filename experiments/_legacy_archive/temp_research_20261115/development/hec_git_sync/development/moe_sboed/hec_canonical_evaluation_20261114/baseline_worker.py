"""Original V2 historical-baseline inference namespace, exact bank and seed schedule."""
import argparse,io,json,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from protocol import BASE,LOCK,sha,load
p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--output',required=True);args=p.parse_args()
from pathlib import Path
out=Path(args.output);inventory=json.loads((BASE/'inputs/HISTORICAL_BASELINES.json').read_text())
row=next(r for r in inventory['rows'] if r['method']==args.method)
assert sha(row['checkpoint'])==row['checkpoint_sha256']
if row.get('arrays_path'):assert sha(row['arrays_path'])==row['arrays_sha256']
sys.path.insert(0,inventory['source_root'])
from src.research.journal_checkpoints import load_checkpoint
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
bank,u,score=load();b=make_benchmark(inventory['physics'])
policy=torch.load(row['checkpoint'],map_location='cpu',weights_only=False) if row.get('loader')=='object' or args.method=='rl_sboed' else load_checkpoint(row['checkpoint'],b)
buf=io.BytesIO();torch.save(policy,buf);buf.seek(0);policy=torch.load(buf,map_location='cuda',weights_only=False).eval()
if hasattr(policy,'stochastic'):policy.stochastic=False
values=[];curves=[]
with torch.no_grad(),torch.device('cuda'):
    neg=torch.tensor(bank['contrasts'],device='cuda',dtype=torch.float64)
    for i in range(0,2000,16):
        data=rollout(b,policy,30,min(16,2000-i),2026110813902+i*97451,0,
            truth=torch.tensor(bank['truth'][i:i+16],device='cuda',dtype=torch.float64),
            noise=torch.tensor(bank['noise'][i:i+16,:,None],device='cuda',dtype=torch.float64),
            method='rl_sboed' if args.method=='rl_sboed' else None)
        result,_=score(b,data,neg,16384,'cuda');curves.append(result.numpy());values.extend(result[:,-1].numpy())
saved=np.load(BASE/f'evaluation/{args.method}_scores.npy');values=np.array(values)
error=float(np.abs(values-saved).max());assert error<1e-9,f'{args.method}: failed historical reproduction {error}'
assert sha(row['checkpoint'])==row['checkpoint_sha256']
np.save(out/f'{args.method}_scores.npy',values);np.save(out/f'{args.method}_curves.npy',np.concatenate(curves))
(out/f'{args.method}_REPRODUCTION.json').write_text(json.dumps(dict(status='PASS',max_abs_error=error,checkpoint_sha256=row['checkpoint_sha256'],saved_scores_sha256=sha(BASE/f'evaluation/{args.method}_scores.npy')),indent=2))
