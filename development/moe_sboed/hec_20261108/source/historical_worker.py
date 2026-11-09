"""Separate native-baseline namespace;our exact frozen scorer and new bank."""
import os,sys,pathlib,json,time,hashlib,types,io
sys.dont_write_bytecode=True
O=pathlib.Path(__file__).resolve().parents[1];inventory=json.loads((O/'inputs/HISTORICAL_BASELINES.json').read_text());sys.path.insert(0,inventory['source_root']);import numpy as np,torch
from src.research.journal_checkpoints import load_checkpoint
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from frozen_evaluator import contrast_scores
method=sys.argv[1];preflight='--preflight' in sys.argv;row=next(r for r in inventory['rows'] if r['method']==method);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert sha(row['checkpoint'])==row['checkpoint_sha256']
if row.get('arrays_path'):assert sha(row['arrays_path'])==row['arrays_sha256']
torch.set_num_threads(2 if preflight else 4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device='cpu' if preflight else 'cuda';beg=time.monotonic();b=make_benchmark(inventory['physics']);policy=torch.load(row['checkpoint'],map_location='cpu',weights_only=False) if row.get('loader')=='object' or method=='rl_sboed' else load_checkpoint(row['checkpoint'],b);buf=io.BytesIO();torch.save(policy,buf);buf.seek(0);policy=torch.load(buf,map_location=device,weights_only=False).eval()
if hasattr(policy,'stochastic'):policy.stochastic=False
if preflight:
 rng=np.random.default_rng(2026110815501);bank=dict(truth=rng.standard_normal((4,4)),noise=rng.standard_normal((4,30)),contrasts=rng.standard_normal((11,4)))
else:bank=np.load(O/'evaluation/development_validation_bank.npz')
scores=[];curves=[]
with torch.no_grad(),torch.device(device):
 b=make_benchmark(inventory['physics']);contrasts=torch.tensor(bank['contrasts'],device=device,dtype=torch.float64)
 for i in range(0,len(bank['truth']),16):
  n=min(16,len(bank['truth'])-i);data=rollout(b,policy,30,n,2026110813902+i*97451,0,truth=torch.tensor(bank['truth'][i:i+n],device=device,dtype=torch.float64),noise=torch.tensor(bank['noise'][i:i+n,:,None],device=device,dtype=torch.float64),method='rl_sboed' if method=='rl_sboed' else None);values,_=contrast_scores(b,data,contrasts,3 if preflight else 16384,device);assert torch.isfinite(values).all();curves.append(values.cpu().numpy());scores.extend(values[:,-1].cpu().numpy())
assert sha(row['checkpoint'])==row['checkpoint_sha256'];out=O/('evidence' if preflight else 'evaluation');out.mkdir(exist_ok=True)
if not preflight:np.save(out/f'{method}_scores.npy',np.array(scores));np.save(out/f'{method}_curves.npy',np.concatenate(curves))
(out/f'{method}_{"preflight" if preflight else "receipt"}.json').write_text(json.dumps(dict(status='PASS' if preflight else 'COMPLETE',method=method,n=len(scores),mean=float(np.mean(scores)),MCSE=float(np.std(scores,ddof=1)/np.sqrt(len(scores))),seconds=time.monotonic()-beg,simulator_observations=len(scores)*30,negative_density_terms=len(scores)*30*len(bank['contrasts']),positive_density_terms=len(scores)*30,checkpoint_sha256=row['checkpoint_sha256'],native_training_status=row['status'],not_compute_matched=True,prior_contrasts=len(bank['contrasts']),positive_included_once=True,training_updates=0),indent=2));print('HISTORICAL_EVALUATION',method,'PASS' if preflight else 'COMPLETE')
