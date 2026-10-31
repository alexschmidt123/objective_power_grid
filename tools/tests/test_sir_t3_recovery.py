"""Regression coverage for final-checkpoint observer schema and native scheduler."""
from pathlib import Path
import copy,torch,pytest
from src.research import training as tr
from src.research.original_protocol import dad_settings
from src.research.compute_budget import ComputeClock
class B:
 calls=0
class P(torch.nn.Module):
 def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor([.2],dtype=torch.float64))
 def forward(self,x):return self.w*x
class Observer:
 compute_clock=None
 def __init__(self,n):self.n=n;self.events=[];self.last=None
 def allow_next_unit(self,p,episodes,u,opt,sch):return u<self.n
 def check(self,p,episodes,u,opt,sch,force=False):
  self.events.append((u,episodes,opt.param_groups[0]['lr']));self.last=(copy.deepcopy(p.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(sch.state_dict()));return u>=self.n
 def finish(self,p):return dict(selected_checkpoint=dict(update=self.events[-1][0],path='fixture'),checkpoint_selection='latest serialized budget eligible')
def test_final_observer_requires_neither_validation_cadence_nor_checkpoint_eig(tmp_path,monkeypatch):
 def draw(b,p,h,batch,seed,L):return dict(info=(-p.w.square()).expand(batch,h))
 monkeypatch.setattr(tr,'rollout',draw)
 monkeypatch.setattr(tr,'select_checkpoint',lambda *a:pytest.fail('Final checkpoint must not select by validation EIG'))
 s=dict(dad_settings(),updates=1001,hidden=64);obs=Observer(1001);p=P();before=p.w.detach().clone()
 p,data=tr.train_spce(B(),'dad',3,s,101,tmp_path,True,None,p,obs)
 assert data['actual_updates']==1001 and data['selected_update']==1001
 assert not torch.equal(before,p.w) and data['validation_spce'] is None
 assert obs.events[0][2]==pytest.approx(5e-5*.98)
 assert obs.events[999][2]==pytest.approx(5e-5*.98)
 assert obs.events[1000][2]==pytest.approx(5e-5*.98**2)
 q=P();q.load_state_dict(obs.last[0]);torch.testing.assert_close(q.w,p.w)
 opt=torch.optim.Adam(q.parameters(),lr=5e-5,betas=(.8,.998));opt.load_state_dict(obs.last[1]);assert int(opt.state[q.w]['step'])==1001
 state=torch.load(tmp_path/'dad.pt',weights_only=False);assert state['selected_update']==1001

def test_validation_mode_still_uses_declared_cadence(tmp_path,monkeypatch):
 monkeypatch.setattr(tr,'rollout',lambda b,p,h,batch,seed,L:dict(info=(-p.w.square()).expand(batch,h)))
 seen=[]
 def score(*a):seen.append(1);return .5
 monkeypatch.setattr(tr,'select_checkpoint',score)
 s=dict(dad_settings(),updates=3,hidden=64,validate_every=2)
 tr.train_spce(B(),'dad',3,s,101,tmp_path,False,None,P())
 assert len(seen)==3 # initial, update2, final3

def test_compute_credit_excludes_validation_but_not_training_or_serialization(tmp_path):
 now=[10.];clock=ComputeClock(tmp_path/'clock.json',14400,credit_seconds=20.343102435814217,now=lambda:now[0]);now[0]+=2
 with clock.exclude('validation'):now[0]+=5
 now[0]+=3;assert clock.spent()==pytest.approx(25.343102435814217)
 assert clock.checkpoint('valid')['eligible'];clock.stop();now[0]+=100;assert clock.spent()==pytest.approx(25.343102435814217)
