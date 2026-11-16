"""One shared fixed preparation, reused identically by both arms."""
import time,copy,numpy as np
from native_common import *
from restore_sg import restore
from hec_policy import HECActor,COST
from soft_policy import prefix_inputs
from specialist_tasks import targets,supervised_loss
CFG=json.loads((O/'CONFIG_LOCKED.json').read_text())
def create_model(a):
 with torch.random.fork_rng(devices=[0]):torch.manual_seed(CFG['seed']);return HECActor(a.actor_net,a.post_approx,a.critic_net,True).cuda()
def prepare(start):
 assert not (O/'preparation/STARTED.json').exists();D=O/'preparation';D.mkdir();COST.clear();put(D/'STARTED.json',dict(time=time.time(),steps=300,training=4096,validation=512));state=restore();a=state['algorithm'];model=create_model(a);rng=np.random.default_rng(CFG['preparation_seed']);idx=rng.choice(1000000,4608,replace=False);t=rng.integers(30,size=4608);ix=torch.tensor(idx,device='cuda');stage=torch.tensor(t,device='cuda');x=prefix_inputs(a._runtime_buffer['ds_hist'][ix],a._runtime_buffer['ys_hist'][ix],stage);gen=torch.Generator(device='cuda').manual_seed(CFG['teacher_seed']);commands=2*torch.rand(4608,20,device='cuda',generator=gen)-1;commands[::2]=0;labels=[[] for k in range(4)];calls=0;tick=time.monotonic()
 with torch.no_grad(),torch.device('cuda'):
  for i in range(0,4608,128):
   sl=slice(i,i+128);_,info=model(x[sl],command_override=commands[sl],collect=True);tt,receipt=targets(model,x[sl],info,gen)
   for k in range(4):labels[k].append(tt[k])
   calls+=receipt['predictive_observations']
 labels=[torch.cat(v) for v in labels];normal=[v[:4096].std(0).clamp_min(.1) for v in labels];baseline=[v[:4096].mean(0) for v in labels];np.savez_compressed(D/'data.npz',x=x.cpu().numpy(),commands=commands.cpu().numpy(),replay_indices=idx,stages=t,**{f'target{k}':v.cpu().numpy() for k,v in enumerate(labels)});params=[p for module in [*model.experts(),model.readouts] for p in module.parameters()];opt=torch.optim.Adam(params,lr=.001);before=[]
 with torch.no_grad():
  _,loss=supervised_loss(model,x[4096:],commands[4096:],[v[4096:] for v in labels],normal);before=[float(v) for v in loss]
 for step in range(300):
  ids=torch.tensor(rng.integers(4096,size=256),device='cuda');opt.zero_grad(set_to_none=True);loss,parts=supervised_loss(model,x[ids],commands[ids],[v[ids] for v in labels],normal);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step()
  if step%50==0:print('PREP',step,[float(v.detach()) for v in parts],flush=True)
 with torch.no_grad():
  _,loss=supervised_loss(model,x[4096:],commands[4096:],[v[4096:] for v in labels],normal);after=[float(v) for v in loss];constant=[float(((v[4096:]-m)/sd).square().mean()) for v,m,sd in zip(labels,baseline,normal)]
 receipt=dict(status='COMPLETE',steps=300,batch=256,training_histories=4096,heldout_histories=512,before_normalized_MSE=before,after_normalized_MSE=after,constant_normalized_MSE=constant,all_improved_vs_initial=all(a<b for a,b in zip(after,before)),all_better_than_constant=all(a<b for a,b in zip(after,constant)),seconds=time.monotonic()-tick,predictive_observations=calls,model_computation_calls=dict(COST),real_simulator_episodes=0,teacher_is_approximate_belief_native_bootstrap=True,hidden_parameters_read=False)
 torch.save(dict(actor_state=model.state_dict(),preparation_optimizer=opt.state_dict(),normalization=normal,constant_predictors=baseline),D/'prepared.pt');put(D/'RESULT.json',receipt);print('PREPARATION_COMPLETE',json.dumps(receipt),flush=True)
 return receipt
