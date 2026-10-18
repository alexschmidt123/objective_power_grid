"""Original DAD encoder/emitter in SIR dimensions; no training jobs."""
import ast,os,json
from pathlib import Path
import torch
from torch import nn
from src.research.sir_dad import SIRSumDAD
from src.research.journal_checkpoints import load_checkpoint
from src.research.benchmarks import SIR

def native_types():
 ref=Path(os.environ.get('E1_UPSTREAM_REFERENCES',str(Path.home()/'Documents/objective_power_grid/experiments/10182026/10182026_e1_original_methods_v2/upstream_references')))
 env=dict(torch=torch,nn=nn)
 for path,names in [(ref/'dad/location_finding.py',['EncoderNetwork','EmitterNetwork']),(ref/'dad/neural/modules.py',['SetEquivariantDesignNetwork'])]:
  tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
 return env

def test_native_sir_dimensions_empty_history_actions_and_gradients():
 torch.manual_seed(101);n=native_types();encoder=n['EncoderNetwork']((1,1),1,256,16).double();emitter=n['EmitterNetwork'](16,(1,1)).double();ref=n['SetEquivariantDesignNetwork'](encoder,emitter,torch.zeros(1,1,dtype=torch.float64))
 p=SIRSumDAD(5)
 with torch.no_grad():
  for a,b in [(p.encoder[0],encoder.linear1),(p.encoder[2],encoder.output_layer),(p.emitter,emitter.linear)]:a.weight.copy_(b.weight);a.bias.copy_(b.bias)
 tokens=torch.randn(3,5,2,dtype=torch.float64);f=torch.zeros(3,16,dtype=torch.float64);f[:,1:].reshape(3,5,3)[...,:2]=tokens
 pairs=[];local=[];native=[]
 for stage in range(5):
  local.append(p(f,stage));native.append(ref(*pairs).reshape(-1,1).expand(3,-1));pairs.append((tokens[:,stage,:1,None],tokens[:,stage,1:2]))
 torch.testing.assert_close(torch.stack(local),torch.stack(native),atol=1e-12,rtol=1e-12)
 torch.stack(local).square().sum().backward();torch.stack(native).square().sum().backward()
 for a,b in [(p.encoder[0],encoder.linear1),(p.encoder[2],encoder.output_layer),(p.emitter,emitter.linear)]:
  for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x.grad,y.grad,atol=1e-10,rtol=1e-10)
 assert 'first_design' not in p.state_dict()

def test_legacy_and_native_sir_checkpoint_predictions_preserved(tmp_path):
 raw=dict(system=dict(name='sir_ode'),sir_ode=dict(integration_steps_per_stage=512),observation=dict(noise_sigma=1.));b=SIR(raw)
 (tmp_path/'run_config.json').write_text(json.dumps(dict(benchmark=raw)));models=tmp_path/'models';models.mkdir()
 f=torch.randn(3,16,dtype=torch.float64)
 for revision in [1,2]:
  p=SIRSumDAD(5,first_design_revision=revision).eval();path=models/f'dad{revision}.pt'
  torch.save(dict(kind='dad',horizon=5,architecture=p.architecture,state_dict=p.state_dict()),path)
  q=load_checkpoint(path,b)
  for stage in range(5):torch.testing.assert_close(p(f,stage),q(f,stage),atol=0,rtol=0)
