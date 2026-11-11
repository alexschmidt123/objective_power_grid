import copy,time,io
from common import *
from joint_policy import JointHEC
import evaluation as ev
import conditional_diagnostics as cond

def main():
 original.setup();lease=original.acquire();start=time.monotonic();state=restore();parent=copy.deepcopy(state['algorithm'].actor_net).eval().requires_grad_(False);model=JointHEC(copy.deepcopy(parent)).cuda().eval().requires_grad_(False);del state;rng=np.random.default_rng(CFG['validation_seed']+1);bank=dict(truth=rng.standard_normal((512,4)),noise=rng.standard_normal((512,30)));u=rng.random((512,30));contrasts=torch.tensor(rng.standard_normal((11,4)),device='cuda');before=digest(model.state_dict());root=O/'evidence/evaluator_fixture';root.mkdir(exist_ok=True);ev.E=root;hist={}
 with torch.no_grad(),torch.device('cuda'):
  data=ev.simulate(model,*[torch.tensor(bank[k],device='cuda') for k in ['truth','noise']],torch.tensor(u,device='cuda'),'candidate',collect=True);hist={k:data[k].cpu().numpy() for k in ['X','features','commands','gates','slots','legacy_features','responses','uncertainty_trace']};hist.update(ds=torch.stack(data['actions'],1).cpu().numpy(),ys=torch.stack(data['observations'],1).cpu().numpy(),truth=data['truth'].cpu().numpy());small={k:v[:4] for k,v in bank.items()};du=u[:4];d=ev.simulate(model,*[torch.tensor(small[k],device='cuda') for k in ['truth','noise']],torch.tensor(du,device='cuda'),'candidate');repeat=ev.simulate(model,*[torch.tensor(small[k],device='cuda') for k in ['truth','noise']],torch.tensor(du,device='cuda'),'candidate');assert all(torch.equal(a,b) for a,b in zip(d['actions'],repeat['actions']));bench=type('B',(),dict(sigma=.5,calls=0))();values,_=ev.contrast_scores(bench,d,contrasts,3,'cuda');literal=[]
  for i in range(4):
   theta=torch.cat([d['truth'][i:i+1],contrasts]);cum=torch.zeros(12,device='cuda',dtype=torch.float64)
   for t in range(30):cum+=-.5*((d['observations'][t][i,0]-mean_signal(theta[None],d['actions'][t][i:i+1,None])[0,0])/.5).square()-math.log(.5*math.sqrt(2*math.pi))
   literal.append(cum[0]-torch.logsumexp(cum,0)+math.log(12))
  torch.testing.assert_close(values[:,-1].to('cuda'),torch.stack(literal),rtol=1e-9,atol=1e-9);assert bench.calls==1320
  for k in range(4):
   g=torch.eye(4,device='cuda')[k][None].expand(30,-1);v=ev.simulate(model,*[torch.tensor(small[s],device='cuda') for s in ['truth','noise']],torch.tensor(du,device='cuda'),'candidate',gates=g);assert all(torch.isfinite(a).all() and bool((a.abs()<=4).all()) for a in v['actions'])
  v=ev.simulate(model,*[torch.tensor(small[s],device='cuda') for s in ['truth','noise']],torch.tensor(du,device='cuda'),'candidate',gates=torch.full((30,4),.25,device='cuda'));assert all(torch.isfinite(a).all() for a in v['actions'])
  # Memory-only parameter state roundtrip;no model checkpoint is modified.
  buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);clone=JointHEC(copy.deepcopy(parent)).cuda().eval().requires_grad_(False);clone.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=False));x=torch.tensor(hist['X'][:4,12],device='cuda');assert torch.equal(model(x),clone(x))
  real_loader=ev.actor_load
  def fixture_loader(method):return (copy.deepcopy(model) if method=='candidate' else copy.deepcopy(parent)),None
  ev.actor_load=fixture_loader
  try:skills=ev.competency(model,hist,time.monotonic()+120)
  finally:ev.actor_load=real_loader
  cond.O=root;(root/'evaluation').mkdir(exist_ok=True);cond.run_conditional(model,hist,time.monotonic()+120)
  for method in ['conventional','vsoed']:
   aa,_=ev.actor_load(method);v=ev.simulate(aa,*[torch.tensor(small[s],device='cuda') for s in ['truth','noise']],torch.tensor(du,device='cuda'),method);assert all(torch.isfinite(a).all() and bool((a.abs()<=4).all()) for a in v['actions'])
 assert digest(model.state_dict())==before;put(O/'evidence/EVALUATION_CHECKS.json',dict(status='PASS',seconds=time.monotonic()-start,literal_contrast_convention=True,repeat_full_trajectory_bitwise=True,all_frozen_ablation_interfaces=True,actor_state_roundtrip_bitwise=True,competency_probe_pipeline_valid=True,models_unchanged=True,training_updates=0));put(O/'evidence/CONDITIONAL_CHECKS.json',dict(status='PASS',complete_conditional_resimulation_pipeline=True,identical_continuation_consistency=True,no_saved_future_replay=True,grouped_native_return_not_exact_conditional_MI=True,training_updates=0));print('HECV21_EVALUATION_AND_CONDITIONAL_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
