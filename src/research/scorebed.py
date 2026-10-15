"""Official JAX ScoreBED training and a verified Torch inference bridge.

The previous small conservative-attention adapter is historical/excluded.
"""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

class OfficialScorePolicy(nn.Module):
    kind='scorebed'
    architecture='official_ScoreBED_DADNet_JAX_weights_Torch_inference'
    def __init__(self,arrays,horizon,action_scale=4.,obs_scale=3.):
        super().__init__();self.horizon=horizon;self.action_scale=action_scale;self.obs_scale=obs_scale
        self.design_bounds=tuple(arrays["common_design_bounds"]) if "common_design_bounds" in arrays else None
        self.outputs_design=True
        self.layers=nn.ParameterDict()
        for section,layer in [('encoder_net','layers_0'),('encoder_net','layers_2'),('emitter_net','layers_0')]:
            prefix=section+'_'+layer
            for name in ['kernel','bias']:
                self.layers[prefix+'_'+name]=nn.Parameter(torch.from_numpy(np.array(arrays[f'{section}/{layer}/{name}'],dtype=np.float32)))
    def linear(self,x,section,layer):
        prefix=section+'_'+layer
        return x@self.layers[prefix+'_kernel']+self.layers[prefix+'_bias']
    def forward(self,features,stage):
        slots=features[:,1:].reshape(len(features),self.horizon,4)
        x=(slots[:,:stage,:2]*self.action_scale).float()
        y=(slots[:,:stage,2:3]*self.obs_scale).float()
        encoding=torch.full((len(features),16),.01,dtype=torch.float32,device=features.device)
        if stage:
            tokens=torch.cat((y,x),-1)
            enc=self.linear(torch.relu(self.linear(tokens,'encoder_net','layers_0')),'encoder_net','layers_2')
            # Match the official chronological accumulation, not a padded sum.
            for t in range(stage):encoding=encoding+enc[:,t]
        raw=self.linear(encoding,'emitter_net','layers_0')
        if self.design_bounds is not None:
            lo,hi=self.design_bounds
            raw=lo+(hi-lo)*raw.sigmoid()
        return raw.double()

def verify_bridge(policy,arrays):
    x=arrays['reference_x'];y=arrays['reference_y'];expected=arrays['reference_actions']
    max_error=0.
    for t in range(policy.horizon):
        f=torch.zeros(len(x),1+4*policy.horizon,dtype=torch.float64)
        slots=f[:,1:].reshape(len(x),policy.horizon,4)
        slots[:,:t,:2]=torch.from_numpy(x[:,:t]).double()/policy.action_scale
        slots[:,:t,2:3]=torch.from_numpy(y[:,:t]).double()/policy.obs_scale
        slots[:,:t,3]=1.
        with torch.no_grad():a=policy(f,t).numpy()
        np.testing.assert_allclose(a,expected[:,t],rtol=2e-5,atol=2e-5)
        max_error=max(max_error,float(np.max(np.abs(a-expected[:,t]))))
    return max_error

def train_scorebed(b,h,s,seed,directory,trajectory_budget=None):
    if b.name!='location_finding':
        raise ValueError('Official ScoreBED integration supports location finding only')
    root=Path(__file__).resolve().parents[2];directory=Path(directory).resolve()
    python=root/'.venv_scorebed/bin/python'
    if not python.is_file():raise RuntimeError('Official ScoreBED isolated environment is missing')
    command=[str(python),str(root/'src/research/scorebed_worker.py'),'--output',str(directory),'--horizon',str(h),'--seed',str(seed),'--budget',str(int(trajectory_budget or 1000000)),'--batch-size',str(int(s.get('scorebed_batch_size',64)))]
    if b.design_bounds is not None:command+=['--design-bounds',*map(str,b.design_bounds)]
    if s['updates']<=2:command.append('--smoke')
    env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled')
    if s['updates']<=2:env['JAX_PLATFORMS']='cpu'
    start=time.perf_counter();subprocess.run(command,cwd=root,env=env,check=True)
    arrays=np.load(directory/'scorebed_official_policy.npz')
    with torch.device('cpu'):
        p=OfficialScorePolicy(arrays,h,b.action_scale,b.obs_scale).eval()
        error=verify_bridge(p,arrays)
    data=json.loads((directory/'scorebed_official_training.json').read_text())
    data.update(training_seconds=time.perf_counter()-start,policy_bridge_max_absolute_error=error,policy_parameters=sum(v.numel() for v in p.parameters()),reported_eig='shared held-out explicit-likelihood evaluator, never score network estimates')
    torch.save(dict(kind='scorebed',architecture=p.architecture,state_dict=p.state_dict(),horizon=h,upstream_commit=data['upstream_commit'],design_bounds=b.design_bounds),directory/'scorebed.pt')
    return p,data
