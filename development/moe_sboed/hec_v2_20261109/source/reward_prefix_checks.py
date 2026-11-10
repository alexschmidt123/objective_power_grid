import torch,math
from native_common import *
from restore_v1 import restore
from prepare_v2 import construct,calibrate
from soft_policy import prefix_inputs
from native_reward import stateless_rewards
from query_response import prospective_opportunity
original.setup();leases=original.acquire();state=restore();a=state['algorithm'];model=construct(a);ds=a._runtime_buffer['ds_hist'][:60];ys=a._runtime_buffer['ys_hist'][:60];t=torch.arange(30,device='cuda').repeat(2);x=prefix_inputs(ds,ys,t)
with torch.no_grad(),torch.device('cuda'):
 calibrate(model,x);_,info=model(x,collect=True);assert torch.isfinite(info['semantic4']).all();assert bool((info['semantic4'][:,:9]>=-1e-6).all());assert torch.equal(info['semantic4'][t==29,:9],torch.zeros_like(info['semantic4'][t==29,:9]));future=torch.arange(30,device='cuda')[None]>=t[:,None];d2=ds.clone();y2=ys.clone();d2[future]=100;y2[future]=-100;assert torch.equal(x,prefix_inputs(d2,y2,t));theta=a._runtime_buffer['params'][:8];r=stateless_rewards(a,ds[:8],ys[:8],theta);terminal=a.post_approx.log_poi_post(29,ds[:8],ys[:8],theta)-a.post_approx.prior.log_poi_prior(theta);torch.testing.assert_close(r.sum(-1),terminal,rtol=1e-5,atol=2e-5)
 w,mu,sd=model.belief.params(x)
 for k in [1,12,29]:
  sl=t==k;net=model.belief.nets[k-1];packed=torch.cat([ds[sl,:k].reshape(-1,2*k),ys[sl,:k].reshape(-1,k)],-1);want=net.weight_net(net.feature_net(packed)).exp();torch.testing.assert_close(w[sl],want,rtol=0,atol=0)
 # Independent unrolled check of predictive Bayes reweighting and future-feature averaging.
 nodes=info['nodes'][:1].double();pw=info['pw'][:1].double();actions=info['probe4'][:1].double();c4=info['commands'][:1,16:].double();stage=torch.tensor([12],device='cuda');actual=prospective_opportunity(nodes,pw,actions,c4,stage)[0];idx=torch.searchsorted(pw[0].cumsum(0),((torch.arange(8,device='cuda',dtype=torch.float64)+.5)/8)).clamp_max(63);th=nodes[0,idx].reshape(8,2,2)
 def signal(ac):return (.1+((th-ac).square().sum(-1)+1e-4).reciprocal().sum(-1)).log()
 v=c4[0,:2]+torch.tensor([0.,1.],dtype=torch.float64,device='cuda');v=v/(v.square().sum()+1e-6).sqrt();angle=math.pi*.5*c4[0,3];v2=torch.stack([v[0]*angle.cos()-v[1]*angle.sin(),v[0]*angle.sin()+v[1]*angle.cos()]);expected=[];firstinfo=[]
 for ac in actions[0]:
  m=signal(ac);loc=m.mean();var=.25+(m-loc).square().mean();futurevalues=torch.zeros(3,dtype=torch.float64,device='cuda');information=torch.zeros((),dtype=torch.float64,device='cuda')
  for offset,weight in [(-math.sqrt(3),1/6),(0.,2/3),(math.sqrt(3),1/6)]:
   yy=loc+var.sqrt()*offset;ww=(-2*(yy-m).square()).softmax(0);center=(ww[:,None,None]*th).sum(0).mean(0)
   for j,k in enumerate([0.,1.,-1.]):
    mm=signal((center+k*v2).clamp(-4,4));pm=(ww*mm).sum();pv=(ww*(mm-pm).square()).sum();futurevalues[j]+=weight*torch.log1p(4*pv)*17/29
   information+=weight*(ww*(ww.clamp_min(1e-38).log()+math.log(8))).sum()
  expected.extend(futurevalues);firstinfo.append(information)
 expected=torch.stack(expected+firstinfo);torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
put(O/'evidence/REWARD_PREFIX_CHECKS.json',dict(status='PASS',training_updates=0,reward_telescopes_full_horizon=True,prefix_belief_packing_exact=True,no_future_history_leakage=True,future_features_zero_at_terminal=True,unrolled_predictive_Bayes_feature_max_error=float((actual-expected).abs().max())));print('REWARD_PREFIX_COUNTERFACTUAL_CHECKS_PASS')
