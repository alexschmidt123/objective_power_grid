"""Shared, explicitly paired terminal sPCE with nested contrast resolution audit."""
import math,time,json,copy,hashlib
from pathlib import Path
import numpy as np,torch
from src.research.engine import rollout,replay_likelihood
from src.research.sir_t3_common import T3SIR

def stats(x):
 x=np.asarray(x,dtype=float);mu=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)));return dict(mean=mu,MC_SE=se,CI95_low=mu-1.96*se,CI95_high=mu+1.96*se,N=len(x))
def cases(raw,seed,N,L,path):
 path=Path(path)
 if path.exists():
  a=np.load(path);truth=torch.from_numpy(a['truth']);noise=torch.from_numpy(a['noise']);bank=torch.from_numpy(a['bank'])
  assert len(truth)==N and len(bank)==L and int(a['seed'])==seed
 else:
  with torch.device('cpu'):
   b=T3SIR(raw);g=torch.Generator().manual_seed(seed);truth=b.sample((N,),g);noise=torch.randn(N,3,1,dtype=torch.float64,generator=g);bank=b.sample((L,),g)
  np.savez(path,truth=truth.numpy(),noise=noise.numpy(),bank=bank.numpy(),seed=seed)
 digest=hashlib.sha256(truth.numpy().tobytes()+noise.numpy().tobytes()+bank.numpy().tobytes()).hexdigest()
 return truth,noise,bank,digest

@torch.no_grad()
def evaluate(raw,policy,seed,N,contrasts,directory,tag,method=None):
 started=time.monotonic();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);counts=sorted(set(contrasts));truth,noise,bank,identity=cases(raw,seed,N,max(counts),directory/f'cases_seed{seed}_N{N}_L{max(counts)}.npz');output={L:[] for L in counts}
 with torch.device('cuda:0'):
  b=T3SIR(raw);p=copy.deepcopy(policy).cuda().eval() if policy is not None else None
  if p is not None:
   for module in p.modules():
    for attr in ['bounds','has_inf']:
     if torch.is_tensor(getattr(module,attr,None)):setattr(module,attr,getattr(module,attr).to('cuda:0'))
  # Fixed: all outer histories share identical actions. Reuse bank states
  # exactly; randomized/adaptive designs use complete independent trajectories.
  fixed=p is not None and getattr(p,'kind',None)=='fixed';bank_means=None
  if fixed:
   probe=rollout(b,p,3,1,seed+44000019,0,truth=truth[:1].cuda(),noise=noise[:1].cuda());bank_means=[]
   for start in range(0,len(bank),16384):
    th=bank[start:start+16384].cuda();state=b.initial((len(th),));means=[]
    for action in probe['actions']:m,state=b.forward(th,state,action.expand(len(th),-1));means.append(m[:,0].cpu())
    bank_means.append(torch.stack(means,1))
   bank_means=torch.cat(bank_means)
  for start in range(0,N,16):
   n=min(16,N-start);r=rollout(b,p,3,n,seed+start*97451,0,truth=truth[start:start+n].cuda(),noise=noise[start:start+n].cuda(),method='random' if method=='random' else None)
   ll,_,_=replay_likelihood(b,r['truth'][:,None],r['actions'],r['observations']);true=ll[:,0];sumc=torch.full_like(true,-torch.inf);stops=set(counts)
   offset=0
   while offset<len(bank):
    end=min(offset+16384,min(x for x in counts if x>offset));chunk=bank[offset:end].cuda()
    if fixed:
     means=bank_means[offset:end].cuda();score=torch.zeros(n,end-offset,dtype=torch.float64)
     for stage,y in enumerate(r['observations']):score+=b.log_likelihood(y[:,None,:],means[None,:,stage,None])
    else:score,_,_=replay_likelihood(b,chunk[None].expand(n,-1,-1),r['actions'],r['observations'])
    sumc=torch.logaddexp(sumc,torch.logsumexp(score,-1));offset=end
    if end in stops:output[end].extend((true-torch.logaddexp(true,sumc)+math.log(end+1)).cpu().tolist())
  torch.cuda.synchronize()
 meta=dict(seed=seed,N=N,contrasts=counts,paired_cases_sha256=identity,seconds=time.monotonic()-started,terminal_full_horizon=True,fixed_exact_deterministic_reuse=fixed,MC_SE_scope='conditional on training seed and contrast bank; contrast bias and training-seed variance excluded')
 arrays={L:np.asarray(x) for L,x in output.items()}
 for L,x in arrays.items():
  if not np.isfinite(x).all():raise FloatingPointError('Nonfinite EIG')
  np.save(directory/f'{tag}_L{L}.npy',x)
 (directory/f'{tag}_metadata.json').write_text(json.dumps(meta,indent=2))
 return arrays,meta
