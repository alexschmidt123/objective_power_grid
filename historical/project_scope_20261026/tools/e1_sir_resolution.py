"""Evaluation-only nested-contrast sensitivity, using shared explicit SIR likelihood."""
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np,torch,yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.research.benchmarks import make_benchmark
from src.research.engine import replay_likelihood
from tools import journal_campaign as j
METHODS=j.METHODS;COUNTS=[16384,65536,262144,1000000]
C=R/'experiments/10192026/10192026_e1_sir_verification'
OLD=Path.home()/'Documents/objective_power_grid/experiments/10182026/10182026_e1_sir_t5_seed101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(method):
 p=OLD/'sir/seed101'/method
 if not p.exists():p=OLD/'worker_ubuntu/sir/seed101'/method
 assert (p/'exit_code').read_text().strip()=='0';return p
@torch.no_grad()
def nested(b,truth,actions,obs,bank,counts,chunk=4096,device='cuda:0'):
 with torch.device(device):
  truth=truth.to(device);actions=[x.to(device) for x in actions];obs=[x.to(device) for x in obs]
  positive=replay_likelihood(b,truth[:,None],actions,obs)[0][:,0]
  total=torch.full_like(positive,-torch.inf);out={};start=0;tic=time.perf_counter()
  for n in counts:
   while start<n:
    stop=min(n,start+chunk);params=bank[start:stop].to(device)[None].expand(len(truth),-1,-1)
    ll=replay_likelihood(b,params,actions,obs)[0]
    total=torch.logaddexp(total,torch.logsumexp(ll,1));start=stop
   torch.cuda.synchronize() if str(device).startswith('cuda') else None
   out[n]=dict(scores=(positive-torch.logaddexp(positive,total)+math.log(n+1)).cpu().numpy(),seconds=time.perf_counter()-tic)
 return out

def prepare():
 assert not C.exists()
 for d in ['reports','metadata','evaluations','logs','figures','plotting','configs','frozen_source']:(C/d).mkdir(parents=True,exist_ok=True)
 oldbank=OLD/'evaluations/development_bank';a=np.load(oldbank/'cases.npz');prefix=np.load(oldbank/'contrasts_resolution.npy')
 raw=yaml.safe_load((R/'configs/e1_sir_native.yaml').read_text());b=make_benchmark(raw)
 g=torch.Generator().manual_seed(1001);b.sample((2048,),g);torch.randn(2048,5,1,generator=g,dtype=torch.float64);check=b.sample((len(prefix),),g).numpy();np.testing.assert_array_equal(check,prefix)
 full=np.concatenate([prefix,b.sample((1000000-len(prefix),),g).numpy()])
 np.save(C/'evaluations/contrasts_nested.npy',full);np.savez(C/'evaluations/cases.npz',truth=a['truth'],noise=a['noise'])
 j.write(C/'metadata/bank_manifest.json',dict(seed=1001,contrast_counts=COUNTS,systems=2048,sha256={p.name:sha(p) for p in (C/'evaluations').iterdir()}))
 entries=[]
 for m in METHODS:
  p=source(m);t=json.loads((p/'models'/f'{m}_training.json').read_text());total=t.get('total_training_episodes',t['generated_training_trajectories']);assert total==(0 if m=='random' else 1000000)
  entries.append(dict(method=m,seed=101,run=str(p),checkpoint_sha256=None if m=='random' else sha(p/'models'/f'{m}.pt'),evaluation_sha256=sha(p/'eval'/f'{m}_seed1001.json'),episodes=total))
 j.write(C/'metadata/seed101_inventory.json',entries)
 (C/'configs/sir.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
 j.write(C/'metadata/resolution_protocol.json',dict(subset_size=128,subset='first128 common cases, then enlarge as needed',counts=COUNTS,maximum_mean_shift_target=.02,maximum_paired_contrast_shift_target=.01,selection='common resolution based on numerical changes, not ranking; report unresolved tail if1M insufficient',training='none; saved trajectories rescored',git_commit=j.subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()))
 print(C)
def run(n):
 torch.set_num_threads(2);bank=torch.from_numpy(np.load(C/'evaluations/contrasts_nested.npy'));raw=yaml.safe_load((C/'configs/sir.yaml').read_text());records=[];arrays={}
 for m in METHODS:
  out=C/'evaluations'/f'{m}_N{n}_nested.npz'
  if out.exists():
   a=np.load(out);scores={k:a[str(k)] for k in COUNTS};seconds=a['seconds']
  else:
   rows=json.loads((source(m)/'eval'/f'{m}_seed1001.json').read_text())[:n]
   scores={k:[] for k in COUNTS};seconds=np.zeros(len(COUNTS))
   for begin in range(0,n,16):
    batch=rows[begin:begin+16];truth=torch.tensor([x['truth'] for x in batch],dtype=torch.float64)
    actions=torch.tensor([x['actions'] for x in batch],dtype=torch.float64).unbind(1);obs=torch.tensor([x['observations'] for x in batch],dtype=torch.float64).unbind(1)
    result=nested(make_benchmark(raw),truth,actions,obs,bank,COUNTS)
    for i,k in enumerate(COUNTS):scores[k].extend(result[k]['scores']);seconds[i]+=result[k]['seconds']
    print(m,begin+len(batch),'/',n,'seconds',seconds[-1],flush=True)
   scores={k:np.asarray(v) for k,v in scores.items()}
   np.testing.assert_allclose(scores[16384],np.asarray([x['spce'] for x in rows]),atol=1e-8,rtol=1e-8)
   np.savez(out,**{str(k):v for k,v in scores.items()},seconds=seconds)
  arrays[m]=scores
  for i,k in enumerate(COUNTS):
   values=scores[k];high=values-scores[COUNTS[-1]];next_diff=values-scores[COUNTS[min(i+1,len(COUNTS)-1)]]
   rec=dict(method=m,training_seed=101,evaluation_seed=1001,systems=n,contrasts=k,eig=float(values.mean()),mc_se=float(values.std(ddof=1)/n**.5),difference_to_next=float(next_diff.mean()),difference_to_highest=float(high.mean()),paired_resolution_se=float(next_diff.std(ddof=1)/n**.5),fraction_within_01_ceiling=float((math.log(k+1)-values<.1).mean()),runtime_seconds=float(seconds[i]),bank_sha256=sha(C/'evaluations/contrasts_nested.npy'),source_commit=j.subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),configuration_sha256=sha(C/'configs/sir.yaml'))
   records.append(rec)
  j.table(C/'reports'/f'sir_contrast_resolution_N{n}.csv',records,list(records[0]))
 pairs=[]
 for a,b in [('moe','dad'),('switchmoe','dad'),('moe','switchmoe'),('resmoe','fixed'),('switchresmoe','resmoe')]:
  for k in COUNTS:
   d=arrays[a][k]-arrays[b][k];se=d.std(ddof=1)/n**.5;pairs.append(dict(first=a,second=b,contrasts=k,systems=n,difference=float(d.mean()),mc_se=float(se),low=float(d.mean()-1.96*se),high=float(d.mean()+1.96*se)))
 j.table(C/'reports'/f'sir_resolution_pairs_N{n}.csv',pairs,list(pairs[0]));j.write(C/'metadata'/f'resolution_N{n}_complete.json',dict(status='complete',methods=10,counts=COUNTS))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--systems',type=int,default=128);a=p.parse_args()
 prepare() if a.action=='prepare' else run(a.systems)
