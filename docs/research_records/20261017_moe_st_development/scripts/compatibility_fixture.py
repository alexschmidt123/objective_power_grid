import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_moe_st_routing_dev_103_105';sys.path.insert(0,str(R/'experiments/10162026/10162026_location_root_cause_v4/source'))
import torch,yaml
torch.set_num_threads(2)
from src.research.moe_family import load_policy
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
raw=yaml.safe_load((C/'original.yaml').read_text());b=make_benchmark(raw);g=torch.Generator().manual_seed(771001);truth=b.sample((8,),g);noise=torch.randn(8,30,1,generator=g,dtype=torch.float64);fixture=dict(truth=truth,noise=noise)
with torch.no_grad():
 for seed in [103,105]:
  path=R/f'experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed{seed}/moe/models/moe.pt';p=load_policy(path,b);record=dict(checkpoint=str(path))
  for relaxed in [False,True]:
   p.set_routing(relaxed=relaxed);r=rollout(b,p,30,8,771001,0,truth=truth,noise=noise);record[str(relaxed)]=r['actions']
  fixture[seed]=record
torch.save(fixture,C/'compatibility_fixture.pt');print('ORIGINAL_FIXTURE_SAVED')
