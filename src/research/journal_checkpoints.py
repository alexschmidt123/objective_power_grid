"""Complete native frozen-policy loading with benchmark-physics identity checks."""
import json
from pathlib import Path
import numpy as np
import torch
from src.research.moe_family import benchmark_signature,BasePolicy,load_policy

def load_checkpoint(path,benchmark):
 from src.research.benchmarks import make_benchmark
 path=Path(path).resolve();saved=torch.load(path,map_location='cpu',weights_only=False)
 cfg=json.loads((path.parent.parent/'run_config.json').read_text());source=make_benchmark(cfg['benchmark'])
 if benchmark_signature(source)!=benchmark_signature(benchmark):raise ValueError('Frozen checkpoint benchmark physics/prior differ')
 kind=saved['kind'];h=saved['horizon']
 if kind in {'moe_sboed_family_v1','strategy_moe_sboed'} or saved.get('architecture','').startswith('spce_only_sum_encoder_dselect'):
  return load_policy(path,benchmark)
 if kind in {'sum_single','sir_sum_single_v1'}:return BasePolicy.from_checkpoint(path,benchmark,h).policy.eval()
 if kind=='dad':
  if benchmark.name=='sir_ode':
   from src.research.sir_dad import SIRSumDAD
   architecture=saved.get('architecture')
   if architecture not in {'DAD_sum_encoder_256_16_SIR_normalized_history','DAD_sum_encoder_256_16_SIR_shared_empty_history'}:raise ValueError('Historical generic SIR policy is not silently relabelled as sum DAD')
   p=SIRSumDAD(h,first_design_revision=2 if architecture=='DAD_sum_encoder_256_16_SIR_normalized_history' else 1)
  else:
   from src.research.dad_location_policy import LocationDAD
   p=LocationDAD(h,benchmark.action_scale,benchmark.obs_scale,first_design_revision=saved.get('policy_settings',{}).get('first_design_revision',1))
 elif kind=='fixed':
  from src.research.policies import Policy
  p=Policy(1+h*(benchmark.action_dim+benchmark.obs_dim+1),benchmark.action_dim,h,'fixed',saved.get('hidden',64))
 elif kind=='rl_sboed':
  from src.research.research_redq import RLPolicy
  if saved.get('architecture')=='published_RL_BOED_source_REDQ_native_v1':
   from src.research.native_rl_reference import NativeRLPolicy
   p=NativeRLPolicy(h)
  else:p=RLPolicy(benchmark,h)
 elif kind=='scorebed':
  arrays=np.load(path.with_name('scorebed_official_policy.npz'))
  if benchmark.name=='sir_ode':
   from src.research.sir_scorebed import SIRScorePolicy
   p=SIRScorePolicy(arrays,h,benchmark)
  else:
   from src.research.scorebed import OfficialScorePolicy
   p=OfficialScorePolicy(arrays,h,benchmark.action_scale,benchmark.obs_scale)
 elif kind=='vsoed':
  from src.research.upstream_vsoed import COMMIT,UpstreamActor,compress_input
  from src.research.vendor.vsoed.utils import Net
  if saved.get('commit')!=COMMIT:raise ValueError('Unknown vsOED upstream revision')
  sir=benchmark.name=='sir_ode'
  expected='upstream_GMM_SIR_ODE' if sir else 'upstream_GMM_I'
  if saved.get('implementation')!=expected:raise ValueError('Unregistered vsOED implementation')
  bounds=[[0.,100.]] if sir else [benchmark.design_bounds or [-float('inf'),float('inf')]]*2
  actor=Net([h+(2 if sir else 3)*(h-1),256,256,256,1 if sir else 2],torch.nn.ReLU,torch.tensor(bounds,dtype=torch.float32),'actor').float()
  actor.load_state_dict(saved['actor_state'],strict=True)
  compression=saved.get('network_input_compression','identity')
  if compression=='asinh':actor.register_forward_pre_hook(compress_input)
  elif compression!='identity':raise ValueError('Unknown vsOED input transformation')
  if sir:
   from src.research.sir_upstream_vsoed import UpstreamSIRActor
   p=UpstreamSIRActor(actor,h,benchmark)
  else:p=UpstreamActor(actor,h,benchmark.action_scale,benchmark.obs_scale)
  return p.eval()
 else:raise ValueError('Unregistered native checkpoint kind '+kind)
 p.load_state_dict(saved['state_dict'],strict=True);return p.eval()
