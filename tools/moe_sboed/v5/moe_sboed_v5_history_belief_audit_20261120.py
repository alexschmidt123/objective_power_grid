"""CPU-only candidate audit;no training job or canonical-bank selection."""
from pathlib import Path
import copy,importlib.util,json,sys,time
M=Path.home()/'Documents/objective_power_grid';HERE=M/'src/research/moe_sboed_v5_history_belief_20261120';R=M/'reports/moe_sboed_v5_history_belief_20261120';O=M/'experiments/location_finding/moe_sboed_v5_history_belief_20261120'
spec=importlib.util.spec_from_file_location('role_previous_v5',M/'src/research/moe_sboed_v5_history_20261119/run.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
sys.path.insert(0,str(HERE));from expert_policy import Actor
from role_evidence import evidence
import torch,numpy as np

def new(parent,arm):
 template=p.StudyMoE(parent,202611150101,'uniform');p.engine.seed_all(202611164101%(2**32));return Actor(template,arm)

def main():
 assert not O.exists() and not R.exists();O.mkdir(parents=True);R.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(2)
 state=p.load_parent('cpu');a=state['algorithm'];parent=a.actor_net;old=p.new_actor(parent,'learned');model=new(parent,'learned');uniform=new(parent,'uniform')
 assert list(old.state_dict())==list(model.state_dict())
 assert p.n.digest(old.state_dict())==p.n.digest(model.state_dict())==p.n.digest(uniform.state_dict())
 assert sum(v.numel() for v in old.parameters())==sum(v.numel() for v in model.parameters())
 data=np.load(M/'experiments/location_finding/moe_sboed_v5_belief_20261119/development_histories.npz');ds=torch.tensor(data['designs'][:2]);ys=torch.tensor(data['observations'][:2]);lib=p.n.load_library()
 x=torch.cat([lib.packed(ds,ys,t) for t in [0,1,5,15,29]])
 model.configure('joint');uniform.configure('joint');old.configure('joint')
 with torch.no_grad():
  torch.testing.assert_close(model(x),old(x),atol=0,rtol=0);torch.testing.assert_close(model(x),uniform(x),atol=0,rtol=0)
  c=model.query(x);views=evidence(c,model.query.inference.belief);assert views.shape==(10,4,14)
  assert torch.equal(views[:2,1],torch.zeros_like(views[:2,1])) and torch.equal(views[:2,3],torch.zeros_like(views[:2,3]))
  poisoned=x.clone()
  for row,t in enumerate(x[:,:30].argmax(-1).tolist()):poisoned[row,30+2*t:88]=123.;poisoned[row,88+t:]=456.
  torch.testing.assert_close(model(x),model(poisoned),atol=0,rtol=0)
  # Evidence depends on observed data at identical stages;no optimality claim.
  alt=x.clone()
  for row,t in enumerate(x[:,:30].argmax(-1).tolist()):
   if t:alt[row,88:88+t]+=.4
  altviews=evidence(model.query(alt),model.query.inference.belief)
  hist_effect=(views-altviews).norm(dim=-1).mean(0);assert (hist_effect>0).all()
 calls=[0]*4;handles=[e.register_forward_hook(lambda module,args,result,k=k:calls.__setitem__(k,calls[k]+1)) for k,e in enumerate(model.experts)]
 _,info=model(x,collect=True)
 for h in handles:h.remove()
 assert calls==[1,1,1,1] and info['z'].shape==(10,4,128) and info['routed'].shape==(10,128)
 op=torch.optim.Adam(model.parameters(),lr=p.cfg()['actor_lr']);critic=a.critic_net.requires_grad_(False)
 for step in range(2):
  op.zero_grad();(-critic(torch.cat([x,model(x)],-1)).mean()).backward();grads=p.gradients(model)
  assert all(grads[f'experts.{k}']>0 for k in range(4)) and grads['router.2']>0
  if step==1:assert grads['router.0']>0 and grads['history_context.token']>0
  op.step()
 with torch.no_grad():
  _,ui=uniform(x,collect=True);assert torch.equal(ui['gates'],torch.full_like(ui['gates'],.25))
  handles=[uniform.router.register_forward_pre_hook(lambda *args:(_ for _ in ()).throw(AssertionError('uniform router executed')))]
  uniform(x)
  for h in handles:h.remove()
 spec=importlib.util.spec_from_file_location('role_no_online_audit',M/'tools/moe_sboed_v5_belief_diagnostics_20261119.py');guard=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guard;spec.loader.exec_module(guard)
 deployed=guard.deployment_audit(model,x)
 p.put(O/'role_evidence_snapshot.json',dict(shape=list(views.shape),values=views.tolist(),observation_history_effect=hist_effect.tolist()))
 result=dict(status='PASS',CPU_only=True,source_sha256={str(f.relative_to(M)):p.sha(f) for f in HERE.glob('*.py')},
  actor_parameters=sum(v.numel() for v in model.parameters()),extra_parameters=0,extra_neural_networks=0,previous_state_keys_and_initial_weights_identical=True,
  initial_actions_exactly_match_previous_V5=True,critic_and_posterior_training_unchanged=True,roles=['current uncertainty','predictive innovation under previous belief','observed information under current belief','observed belief change'],
  evidence_shape=[10,4,14],expert_execution_calls=calls,observed_history_changes_role_evidence=hist_effect.tolist(),future_slots_ignored=True,
  no_observation_likelihood_double_counting=True,gradient_norms_after_disposable_critic_steps=grads,deployment=deployed,
  full_training_started=False,EIG_evaluations=0,scientific_status='Implemented untrained candidate;no claim of useful specialization,routing EIG advantage or superiority',seconds=time.monotonic()-start)
 p.put(R/'IMPLEMENTATION_AUDIT.json',result);print(json.dumps({k:result[k] for k in ['status','actor_parameters','extra_parameters','extra_neural_networks','seconds']},indent=2))
if __name__=='__main__':main()
