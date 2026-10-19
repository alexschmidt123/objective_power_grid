"""CPU-only fidelity decision regressions; never launches training."""
import json, sys, io
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.research.baseline_fidelity import scorebed_reported_params, scorebed_restart_validation, rl_optimizer_epoch_completed
from src.research.dad_location_policy import LocationDAD
rows=[]
def check(name, fn):
 fn(); rows.append(dict(test=name,status="PASS"))
def dad_first():
 torch.manual_seed(7)
 a=LocationDAD(2,first_design_revision=1); b=LocationDAD(2,first_design_revision=2)
 b.encoder.load_state_dict(a.encoder.state_dict());b.emitter.load_state_dict(a.emitter.state_dict())
 # Additive shared-emitter invariant in pre-bounded coordinates.
 x,y=torch.randn(2,16,dtype=torch.float64);zero=torch.zeros_like(x)
 gap=a.emitter(x+y)-a.emitter(x)-a.emitter(y)+a.emitter(zero)
 torch.testing.assert_close(gap,torch.zeros_like(gap),atol=1e-14,rtol=0)
 historical_gap=b.emitter(x+y)-b.emitter(x)-b.emitter(y)+b.first_design
 assert historical_gap.abs().max()>1e-3
 features=torch.zeros(1,9,dtype=torch.float64)
 a(features,0).sum().backward();b(features,0).sum().backward()
 assert a.emitter.bias.grad is not None and b.emitter.bias.grad is None
 assert b.first_design.grad is not None
 assert "first_design" not in a.state_dict()
def ema():
 raw,averaged=object(),object();state=SimpleNamespace(params=raw,params_ema=averaged)
 assert scorebed_reported_params(state,True) is averaged
 assert scorebed_reported_params(state,False) is raw
 cfg=scorebed_restart_validation({"validation":{"seed":771001}})
 assert cfg['contrasts']==1000000 and cfg['seed']==771001
 for seed in [1001,8831001]:
  try:scorebed_restart_validation({'validation':{'seed':seed}})
  except ValueError:pass
  else:raise AssertionError('Selection evaluation leakage')
def budget():
 init=10000+4+10;score=692990;policy=1000000-init-score
 assert init+score+policy==1000000 and policy==296996
 allocations=[policy//5+(i<policy%5) for i in range(5)]
 assert sum(allocations)==policy and min(allocations)>0
 assert not rl_optimizer_epoch_completed(99000,100000)
 assert rl_optimizer_epoch_completed(102000,100000)
def serialization():
 from src.research.native_rl_reference import NativeRLPolicy
 for ctor in [lambda:LocationDAD(2,first_design_revision=1),lambda:NativeRLPolicy(2)]:
  p=ctor().eval();f=torch.zeros(2,9,dtype=torch.float64)
  stream=io.BytesIO();torch.save(p.state_dict(),stream);stream.seek(0);q=ctor().eval();q.load_state_dict(torch.load(stream,weights_only=True))
  if isinstance(p,NativeRLPolicy):
   a=p.sample(f,deterministic=True)[0];b=q.sample(f,deterministic=True)[0]
  else:a=p(f,0);b=q(f,0)
  torch.testing.assert_close(a,b,atol=0,rtol=0)
def native_launcher_dispatch():
 import yaml
 config=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/e1_location_original_methods.yaml').read_text())
 assert config['research']['journal_protocol']['version']=='validation_convergence_v1'
 assert config['research']['journal_protocol']['checkpoint_selection_mode']=='final_fixed_budget'
 assert config['research']['journal_protocol']['method_settings']['scorebed']['scorebed_restarts']==1
check('E1_existing_journal_launcher_dispatch',native_launcher_dispatch)
check('DAD_and_native_RL_checkpoint_roundtrip',serialization)
check('DAD_independent_first_design_not_equivalent',dad_first)
check('ScoreBED_native_EMA_export_and_independent_restart_selection',ema)
check('ScoreBED_exact_budget_and_RL_entropy_warmup',budget)
print(json.dumps({'status':'PASS','tests':rows},indent=2))
