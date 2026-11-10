import copy,time,numpy as np,torch
from native_common import *
from restore_v1 import restore
from v2_policy import V2Actor
from soft_policy import prefix_inputs
from v2_tasks import labels,supervised
from hec_policy import COST
CFG=json.loads((O/'CONFIG_LOCKED.json').read_text())
def construct(a,stage=False):return V2Actor(copy.deepcopy(a.actor_net),stage,CFG['new_adapter_seed']).cuda()
@torch.no_grad()
def calibrate(model,x,initial=True):
 t=x[:,:30].argmax(-1);assert torch.bincount(t,minlength=30).min()>0
 if initial:
  w,mu,sd=model.belief.params(x);a0=torch.cat([model.base(y) for y in x.split(256)]);_,ctx=model.context(x,w,mu,sd,a0)
  for stage in range(30):model.stage_context[stage]=ctx[t==stage].mean(0)
  ref=model.commander(model.stage_context);model.reference_logits.copy_(ref);model.stage_logits.copy_(ref)
 old=model.stage_only;model.stage_only=True;cs=[];rs=[]
 for sl in range(0,len(x),128):
  _,i=model(x[sl:sl+128],collect=True);cs.append(torch.cat([i['history'],i['legacy_features'].flatten(1)],-1));rs.append(i['responses'].flatten(1))
 cs=torch.cat(cs);rs=torch.cat(rs)
 for stage in range(30):model.stage_coord_context[stage]=cs[t==stage].mean(0);model.stage_response_context[stage]=rs[t==stage].mean(0)
 model.stage_only=old

def prepare():
 D=O/'preparation';D.mkdir();state=restore();a=state['algorithm'];reference=copy.deepcopy(a.actor_net).eval().requires_grad_(False);model=construct(a);rng=np.random.default_rng(CFG['preparation_seed']);idx=rng.choice(1000000,4608,replace=False);t=rng.integers(30,size=4608);ids=torch.tensor(idx,device='cuda');stage=torch.tensor(t,device='cuda');x=prefix_inputs(a._runtime_buffer['ds_hist'][ids],a._runtime_buffer['ys_hist'][ids],stage);calibrate(model,x[:4096]);gen=torch.Generator(device='cuda').manual_seed(CFG['teacher_seed']);commands=x.new_zeros(4608,20)
 with torch.no_grad():
  for i in range(0,4608,128):commands[i:i+128]=model(x[i:i+128],collect=True)[1]['commands']
  commands[1::4]=(model.stage_logits+model.commander(model.stage_context)-model.reference_logits).tanh()[stage[1::4]];commands[2::4]=0;commands[3::4]=1.8*torch.rand(len(commands[3::4]),20,device='cuda',generator=gen)-.9
 values=[[] for _ in range(4)];observations=0;tick=time.monotonic();COST.clear()
 with torch.no_grad(),torch.device('cuda'):
  for i in range(0,len(x),128):
   sl=slice(i,i+128);_,info=model(x[sl],command_override=commands[sl],collect=True);tt,r=labels(a,reference,model,x[sl],info,gen,4)
   for k in range(4):values[k].append(tt[k])
   observations+=r['predictive_observations']
 values=[torch.cat(v) for v in values];scale=[v[:4096].std(0).clamp_min(.1) for v in values];params=[p for m in [model.adapter2,model.adapter4,model.response_readouts] for p in m.parameters()];opt=torch.optim.Adam(params,lr=.001)
 with torch.no_grad():before=[float(v) for v in supervised(model,x[4096:],commands[4096:],[v[4096:] for v in values],scale)[1]]
 for n in range(300):
  ix=torch.tensor(rng.integers(4096,size=256),device='cuda');opt.zero_grad(set_to_none=True);loss,ls=supervised(model,x[ix],commands[ix],[v[ix] for v in values],scale);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step()
  if n%50==0:print('PREP',n,[float(v.detach()) for v in ls],flush=True)
 with torch.no_grad():after=[float(v) for v in supervised(model,x[4096:],commands[4096:],[v[4096:] for v in values],scale)[1]]
 calibrate(model,x[:4096],False)
 with torch.no_grad():
  before_action=reference(x[4096:]);after_action=model(x[4096:]);torch.testing.assert_close(after_action,before_action,rtol=1e-6,atol=1e-6);assert torch.isfinite(after_action).all()
 np.savez_compressed(D/'data.npz',x=x.cpu().numpy(),commands=commands.cpu().numpy(),replay_indices=idx,stages=t,**{f'target{k}':v.cpu().numpy() for k,v in enumerate(values)});torch.save(dict(actor_state=model.state_dict(),normalization=scale,preparation_optimizer=opt.state_dict(),preparation_rng=rng.bit_generator.state,teacher_generator_state=gen.get_state()),D/'prepared.pt');put(D/'RESULT.json',dict(status='COMPLETE',steps=300,batch=256,training_cases=4096,validation_cases=512,E2_E4_loss_before=before,E2_E4_loss_after=after,seconds=time.monotonic()-tick,predictive_observations=observations,new_real_episodes=0,counts=dict(COST),new_losses_only_preparation=True,latent_samples='conditional frozen approximate belief;not recorded hidden parameters',E4='second-step information and tail-value contrasts;notexactconditionalEIG'));print('PREPARATION_COMPLETE',flush=True)
