"""Official ScoreBED score training/EIG gradient on SIR; CPU Torch policy bridge."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np,torch
from torch import nn
class SIRScorePolicy(nn.Module):
 kind='scorebed';outputs_design=True;architecture='official_ScoreBED_DADNet_SIR_ODE_feasible_time_adapter'
 def __init__(self,arrays,horizon,b):
  super().__init__();self.horizon=horizon;self.end=b.end;self.gap=b.gap;self.population=b.population
  self.layers=nn.ParameterDict()
  for section,layer in [('encoder_net','layers_0'),('encoder_net','layers_2'),('emitter_net','layers_0')]:
   for name in ['kernel','bias']:
    self.layers[section+'_'+layer+'_'+name]=nn.Parameter(torch.from_numpy(np.array(arrays[f'{section}/{layer}/{name}'],dtype=np.float32)))
 def linear(self,x,section,layer):
  p=section+'_'+layer
  return x@self.layers[p+'_kernel']+self.layers[p+'_bias']
 def forward(self,features,stage):
  slots=features[:,1:].reshape(len(features),self.horizon,3)
  enc=torch.full((len(features),16),.01,dtype=torch.float32,device=features.device)
  for t in range(stage):
   token=torch.cat((slots[:,t,1:2],slots[:,t,:1]),-1).float()
   enc=enc+self.linear(torch.relu(self.linear(token,'encoder_net','layers_0')),'encoder_net','layers_2')
  raw=self.linear(enc,'emitter_net','layers_0')[:,0]
  previous=slots[:,stage-1,0]*self.end if stage else features.new_zeros(len(features))
  lo=previous+self.gap;hi=self.end-(self.horizon-stage-1)*self.gap
  return (lo+raw.double().sigmoid()*(hi-lo))[:,None]

def verify_bridge(p,arrays):
 max_error=0.
 for t in range(p.horizon):
  f=torch.zeros(len(arrays['reference_x']),1+3*p.horizon,dtype=torch.float64)
  slots=f[:,1:].reshape(len(f),p.horizon,3)
  slots[:,:t,:1]=torch.from_numpy(arrays['reference_x'][:,:t]).double()/p.end
  slots[:,:t,1:2]=torch.from_numpy(arrays['reference_y'][:,:t]).double()/p.population
  slots[:,:t,2]=1.
  with torch.no_grad():actual=p(f,t).numpy()
  expected=arrays['reference_actions'][:,t]
  np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
  max_error=max(max_error,float(np.abs(actual-expected).max()))
 return max_error

def train_scorebed(b,h,s,seed,directory,trajectory_budget=None):
 root=Path(__file__).resolve().parents[2];directory=Path(directory).resolve()
 config=dict(population=b.population,i0=b.i0,t_end=b.end,min_spacing=b.gap,integration_steps_per_stage=b.steps,beta_mean=float(b.log_mean[0].exp()),gamma_mean=float(b.log_mean[1].exp()),log_std=b.log_std,sigma=b.sigma)
 config_path=directory/'scorebed_sir_configuration.json';config_path.write_text(json.dumps(config,indent=2))
 command=[str(root/'.venv_scorebed/bin/python'),str(root/'src/research/sir_scorebed_worker.py'),'--output',str(directory),'--horizon',str(h),'--seed',str(seed),'--budget',str(int(trajectory_budget or 1000000)),'--batch-size',str(s.get('scorebed_batch_size',64)),'--sir-config',str(config_path)]
 env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled')
 if s['updates']<=2:command+=['--smoke'];env['JAX_PLATFORMS']='cpu'
 start=time.perf_counter();subprocess.run(command,cwd=root,env=env,check=True)
 arrays=np.load(directory/'scorebed_official_policy.npz')
 with torch.device('cpu'):
  policy=SIRScorePolicy(arrays,h,b).eval();error=verify_bridge(policy,arrays)
 data=json.loads((directory/'scorebed_official_training.json').read_text())
 data.update(training_seconds=time.perf_counter()-start,policy_bridge_max_absolute_error=error,policy_parameters=sum(p.numel() for p in policy.parameters()),implementation='Official ScoreBED method with independently verified deterministic SIR ODE adapter; not original published SIR results')
 torch.save(dict(kind='scorebed',architecture=policy.architecture,horizon=h,state_dict=policy.state_dict(),sir_configuration=config),directory/'scorebed.pt')
 return policy,data
