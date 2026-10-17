import importlib.util,numpy as np,json,pathlib,os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
s=importlib.util.spec_from_file_location('diagnosis',str(pathlib.Path(__file__).with_name('scorebed_training_diagnosis.py')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(141);bank=rng.normal(size=(128,4));x=rng.normal(size=(30,2));theta=bank[0].reshape(2,2);mu=np.log(.1+(1/(1e-4+((x[:,None]-theta[None])**2).sum(-1))).sum(-1));y=(mu+.5*rng.normal(size=30))[:,None]
def marginal(z):
 yy=z[:,:1];xx=z[:,1:];diff=xx[None,:,None]-bank.reshape(-1,2,2)[:,None];signal=.1+(1/(1e-4+(diff**2).sum(-1))).sum(-1);ll=-.5*((yy[None,:,0]-np.log(signal))**2).sum(-1)/.25;hi=ll.max();return hi+np.log(np.exp(ll-hi).mean())
ref=np.array(m.reference(y,x,bank,(32,64,128))['128']['score']);z=np.concatenate([y,x],-1);errors=[]
for stage in [0,7,29]:
 for k in range(3):
  a=z.copy();b=z.copy();a[stage,k]+=1e-5;b[stage,k]-=1e-5;fd=(marginal(a)-marginal(b))/(2e-5);errors.append(abs(fd-ref[stage,k]))
assert max(errors)<1e-5,errors
out=pathlib.Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10162026/10162026_location_stability_correction/scorebed_training_diagnosis');out.mkdir(exist_ok=True);(out/'reference_correctness.json').write_text(json.dumps(dict(status='passed',finite_difference_coordinates=9,max_absolute_error=max(errors),reference='Direct finite-sample marginal likelihood gradient, same particles')));print('REFERENCE_GRADIENT_TEST_PASSED',max(errors))
