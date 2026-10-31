"""Final bounded export/device/scoring checks inside the single readiness cap."""
import json,math
from pathlib import Path
import numpy as np,torch,yaml
from tools.sir_seed101_campaign import move_plain_tensors,write
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout,replay_likelihood
from src.research.sir_evaluation import contrast_scores
from src.research.journal_checkpoints import load_checkpoint
from src.research.sir_scorebed_gru import SIRScoreGRUPolicy
from src.research.sir_scorebed import verify_bridge
R=Path(__file__).resolve().parents[1];D=R/'scratch_runs/10272026/10272026_sir_fourhour_readiness';N=D/'native_timings_import_repair'
raw=yaml.safe_load((R/'configs/sir_seed101_fourhour.yaml').read_text());write(N/'run_config.json',dict(benchmark=raw,horizon=5))
torch.set_num_threads(2);records=[]
with torch.device('cpu'):
 b=make_benchmark(raw);g=torch.Generator().manual_seed(97402831);truth=b.sample((4,),g);noise=torch.randn(4,5,1,generator=g,dtype=torch.float64);bank=b.sample((8,),g)
for method in ('random','dad','fixed','rl_sboed','vsoed','scorebed'):
 if method=='random':p=None
 elif method=='scorebed':
  arrays=np.load(N/'scorebed/scorebed_official_policy.npz');p=SIRScoreGRUPolicy(arrays,5,b).eval();bridge=verify_bridge(p,arrays)
 elif method=='rl_sboed':
  from src.research.sir_native_rl import NativeSIRRLPolicy
  p=NativeSIRRLPolicy(5,b);saved=torch.load(N/'rl_sboed/timing_policy.pt',weights_only=False,map_location='cpu');p.load_state_dict(saved['state_dict'])
 else:p=load_checkpoint(N/method/(method+'.pt'),b)
 if p is not None:p=p.cuda().eval();move_plain_tensors(p,'cuda:0')
 with torch.device('cuda:0'),torch.no_grad():
  b=make_benchmark(raw);r=rollout(b,p,5,4,97402832,0,truth=truth.cuda(),noise=noise.cuda(),method=method)
  info,_=contrast_scores(b,r,bank,3)
  particles=torch.cat((truth.cuda()[:,None],bank.cuda()[None].expand(4,-1,-1)),1)
  ll=replay_likelihood(b,particles,r['actions'],r['observations'])[0]
  direct=ll[:,0]-torch.logsumexp(ll,1)+math.log(9)
  error=float((direct.cpu()-info[:,-1]).abs().max());assert error<1e-10,(method,error)
  action=torch.stack(r['actions'],1)[...,0]
  assert torch.isfinite(info).all() and torch.all(action[:,1:]-action[:,:-1]>=.1-1e-9)
 if p is not None:
  p.cpu();move_plain_tensors(p,'cpu')
  with torch.no_grad():assert torch.isfinite(p(r['features'][2].cpu(),2)).all()
 records.append(dict(method=method,streamed_direct_score_max_error=error,cpu_and_gpu_export_forward='PASS',chronology='PASS',scorebed_bridge_error=bridge if method=='scorebed' else None))
write(D/'checkpoint_device_scoring.json',dict(status='PASS',workability_only=True,records=records,primary_training=False))
print(json.dumps(records,indent=2),flush=True)
