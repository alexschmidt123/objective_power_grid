"""Benchmark adapters and pinned baseline provenance, without costly training."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from src.research.benchmarks import make_benchmark,SIR
from src.research.engine import generator,rollout
from src.research.policies import history_features
from src.research.journal_checkpoints import load_checkpoint
from src.research.sir_upstream_vsoed import SIRNativeModel,UpstreamSIRActor,physical_time

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('vendor,key',[('vsoed','unmodified_sha256'),('scorebed','sha256')])
def test_pinned_vendor_hashes(vendor,key):
    folder=ROOT/'src/research/vendor'/vendor
    provenance=json.loads((folder/'PROVENANCE.json').read_text())
    for file,digest in provenance[key].items():
        assert hashlib.sha256((folder/file).read_bytes()).hexdigest()==digest,file

def test_native_sir_simulator_matches_exact_benchmark():
    torch.set_num_threads(1)
    b=SIR({});m=SIRNativeModel(b,5)
    phi=torch.tensor([[0.,0.],[.2,-.3]],dtype=torch.float64)
    theta=b.inverse(phi);state=b.initial((2,));previous=torch.zeros(2,1,dtype=torch.float64)
    for stage in range(5):
        percent=torch.tensor([[10.],[25.]],dtype=torch.float64)
        action=physical_time(stage,percent,previous,5,b.end,b.gap)
        expected,state=b.forward(theta,state,action)
        actual=m.deterministic(stage,phi,percent)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(m.xp_f(stage,None,percent,None),action.float(),rtol=0,atol=0)
        previous=action

def test_sir_native_history_uses_observations_and_percentages():
    b=SIR({});h=5
    class Capture(torch.nn.Module):
        def forward(self,x):self.received=x.clone();return x.new_full((len(x),1),30.)
    actor=Capture();p=UpstreamSIRActor(actor,h,b)
    actions=[];ys=[];previous=torch.zeros(2,1,dtype=torch.float64)
    for stage in range(h):
        f=history_features(b,actions,ys,2,h)
        action=p(f,stage);expected=physical_time(stage,torch.full((2,1),30.),previous,h,b.end,b.gap)
        torch.testing.assert_close(action,expected,rtol=1e-12,atol=1e-12)
        x=actor.received
        torch.testing.assert_close(x[:,h:h+stage],torch.full((2,stage),30.),rtol=1e-5,atol=1e-5)
        if stage:torch.testing.assert_close(x[:,h+h-1:h+h-1+stage],torch.stack(ys,1).squeeze(-1).float())
        actions.append(action);ys.append(torch.full((2,1),float(stage+1),dtype=torch.float64));previous=action

@pytest.mark.parametrize('benchmark,smoke',[('location_finding','location_vsoed_smoke'),('sir_ode','sir_native_vsoed_smoke')])
def test_native_vsOED_checkpoint_reload_and_rollout(benchmark,smoke):
    campaign=ROOT/'experiments/10152026/10152026_journal_location_sir/readiness'/smoke
    if not (campaign/'models/vsoed.pt').exists():pytest.skip('Local native smoke artifact unavailable')
    cfg=json.loads((campaign/'run_config.json').read_text())
    b=make_benchmark(cfg['benchmark']);p=load_checkpoint(campaign/'models/vsoed.pt',b)
    saved=torch.load(campaign/'models/vsoed.pt',map_location='cpu',weights_only=False)
    torch.testing.assert_close(p.actor.state_dict(),saved['actor_state'],rtol=0,atol=0)
    r=rollout(b,p,saved['horizon'],2,991,2)
    assert torch.isfinite(r['info']).all()
    a=torch.stack(r['actions'],1)
    if benchmark=='location_finding':assert ((a>=-4)&(a<=4)).all()
    else:
        assert (a[:,0]>=b.gap).all() and (a[:,-1]<=b.end).all()
        assert (a[:,1:]-a[:,:-1]>=b.gap-1e-9).all()

@pytest.mark.parametrize('config',['location_finding_eig.yaml','sir_ode_research.yaml'])
def test_random_actions_obey_common_constraints(config):
    raw=yaml.safe_load((ROOT/'configs'/config).read_text());b=make_benchmark(raw);h=raw['experiment']['step_number']
    r=rollout(b,None,h,4,992,2,method='random');a=torch.stack(r['actions'],1)
    assert torch.isfinite(r['info']).all()
    if b.name=='location_finding':assert ((a>=-4)&(a<=4)).all()
    else:assert (a[:,1:]-a[:,:-1]>=b.gap-1e-9).all() and (a[:,-1]<=b.end).all()
