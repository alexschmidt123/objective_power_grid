"""SIR-only corrective adaptations; native objectives/vendor files remain unchanged."""
import copy
import json
import torch
import pytest
from src.research.benchmarks import SIR
from src.research.sir_upstream_vsoed import SIRInputUnits,install_actor_input_units,UpstreamSIRActor,physical_time
from src.research.vendor.vsoed.utils import Net
from src.research.policies import history_features
from src.research.journal_checkpoints import load_checkpoint
from pathlib import Path

@pytest.mark.parametrize('critic',[False,True])
def test_units_are_exact_invertible_and_differentiable(critic):
 x=torch.arange(1,14+int(critic),dtype=torch.float64).reshape(1,-1).requires_grad_()
 scale=x.new_tensor([1.]*5+[100.]*4+[500.]*4+([100.] if critic else []))
 y=SIRInputUnits(5,4,500,critic)(None,(x,))[0]
 torch.testing.assert_close(y*scale,x,rtol=0,atol=1e-14)
 torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],1/scale[None,:],rtol=0,atol=0)

def test_native_train_eval_history_agree():
 from src.research.vendor.vsoed import PGvsOED
 b=SIR({});h=5
 actor=Net([13,256,256,256,1],torch.nn.ReLU,torch.tensor([[0.,100.]]),'actor').float()
 install_actor_input_units(actor,h,500)
 wrapper=UpstreamSIRActor(actor,h,b);actions=[];obs=[];ds=[]
 for stage in range(h):
  f=history_features(b,actions,obs,2,h)
  features=torch.zeros(2,13);features[:,stage]=1
  if stage:
   features[:,5:5+stage]=torch.stack(ds,1).squeeze(-1)
   features[:,9:9+stage]=torch.stack(obs,1).squeeze(-1).float()
  percent=actor(features)
  previous=actions[-1] if stage else torch.zeros(2,1,dtype=torch.float64)
  expected=physical_time(stage,percent,previous,h,100,.1)
  torch.testing.assert_close(wrapper(f,stage),expected,rtol=1e-6,atol=1e-6)
  actions.append(expected.detach());obs.append(torch.full((2,1),30.+stage,dtype=torch.float64));ds.append(percent.detach())

def test_corrected_checkpoint_roundtrip(tmp_path):
 b=SIR({});actor=Net([13,256,256,256,1],torch.nn.ReLU,torch.tensor([[0.,100.]]),'actor').float()
 install_actor_input_units(actor,5,500);wrapper=UpstreamSIRActor(actor,5,b)
 models=tmp_path/'models';models.mkdir()
 (tmp_path/'run_config.json').write_text(json.dumps({'benchmark':{'system':{'name':'sir_ode'}}}))
 from src.research.upstream_vsoed import COMMIT
 p=models/'vsoed.pt';torch.save(dict(kind='vsoed',horizon=5,commit=COMMIT,implementation='upstream_GMM_SIR_ODE',actor_state=actor.state_dict(),network_input_compression='sir_units',sir_input_population=500),p)
 restored=load_checkpoint(p,b)
 f=history_features(b,[],[],2,5)
 torch.testing.assert_close(restored(f,0),wrapper(f,0),rtol=0,atol=0)

def test_scaling_avoids_initial_saturation_on_diagnostic_histories():
 torch.manual_seed(101);actor=Net([13,256,256,256,1],torch.nn.ReLU,torch.tensor([[0.,100.]]),'actor').float()
 x=torch.zeros(1024,13);x[:,4]=1;x[:,5:9]=torch.rand(1024,4)*100;x[:,9:]=torch.rand(1024,4)*500
 raw=actor(x);install_actor_input_units(actor,5,500);corrected=actor(x)
 assert ((raw>.99*100)|(raw<.01*100)).float().mean()>.1
 assert ((corrected>.99*100)|(corrected<.01*100)).sum()==0
