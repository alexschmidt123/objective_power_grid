from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path('/home/grads/g/g.lin/Documents/objective_power_grid')
CAMPAIGN=ROOT/'experiments/09162026/09162026_ieee9_moe_endpoint_train101_eval1001'
sys.path.insert(0,str(CAMPAIGN/'source'))
from src.config import load_config
from src.objectives.eig.continuous_eig import OnlineEIG
from src.objectives.eig.continuous_moe import ContinuousMoEPolicy
from src.domains.swing.continuous_rocof import EndpointRocofObserver

class Decision(torch.nn.Module):
    def __init__(self,policy,kind):super().__init__();self.policy=policy;self.kind=kind
    def forward(self,x,stage):
        d,logits,chosen,means=self.policy.components(x)
        if self.kind=='mean':return d,torch.zeros(len(x))
        if self.kind=='top_gate':z=means[:,0]
        elif self.kind=='density_mode':
            grid=torch.linspace(-8,8,513)[:,None].expand(-1,len(x))
            z=grid[d.log_prob(grid).argmax(0),torch.arange(len(x))]
        else:
            parameters=self.policy.experts[int(self.kind[-1])](self.policy.encoder(x))
            z=4*torch.tanh(parameters[:,0]/4)
        return torch.distributions.Normal(z,torch.ones_like(z)),torch.zeros(len(x))

report={}
for horizon in (3,4,5):
    run=CAMPAIGN/f'T{horizon}/run'
    records=json.loads((run/'models/moe_sboed_training.json').read_text())
    diag=json.loads((run/'models/moe_sboed_training_diagnostics.json').read_text())
    rows=json.loads((run/'rollouts.json').read_text())
    stages=[]
    for stage in range(horizon):
        traces=[r['routing_trace'][stage] for r in rows]
        ids=np.array([t['selected_experts'][0] for t in traces])
        weights=np.array([t['selected_weights'][0] for t in traces])
        means=np.array([t['selected_expert_latent_means'][0] for t in traces])
        total=np.zeros(4)
        for selected,w in zip(ids,weights):np.add.at(total,selected,w)
        stages.append({'stage':stage,'selection_counts':np.bincount(ids.ravel(),minlength=4).tolist(),
            'mean_weight_per_expert':(total/len(rows)).tolist(),
            'distinct_pairs':len(set(tuple(sorted(x)) for x in ids)),
            'mean_latent_expert_gap':float(np.abs(np.diff(means,axis=1)).mean()),
            'mean_largest_weight':float(weights.max(1).mean())})
    cfg=load_config(CAMPAIGN/'source/configs/ieee9_eig.yaml')
    observer=EndpointRocofObserver(cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.5)
    e=OnlineEIG(cfg,observer,horizon=horizon,sigma=.005,contrasts=128,min_separation=.01)
    p=ContinuousMoEPolicy(horizon,1)
    payload=torch.load(run/'models/moe_sboed.pth',weights_only=False,map_location='cpu')
    p.load_state_dict(payload['state_dict']);p.eval()
    val={}
    with torch.no_grad():
        for name in ['mean','top_gate','density_mode','expert0','expert1','expert2','expert3']:
            result=e.rollout(Decision(p,name),np.random.default_rng(900101),128,stochastic=False)
            val[name]=float(result['info'][:,-1].mean())
    assert abs(val['mean']-diag['best_validation_utility'])<1e-6
    report[str(horizon)]={'best_update':diag['best_update'],'validation_history':[
        {'update':r['update'],'eig':r['validation_utility'],'router':r['ppo'].get('dense_router_mean_probabilities')}
        for r in records], 'test_routing':stages,'validation_only_decision_interventions':val,
        'late_validation_range':[min(r['validation_utility'] for r in records[-10:]),max(r['validation_utility'] for r in records[-10:])],
        'training_selected_counts_best':next(r['routing'] for r in records if r['update']==diag['best_update'])}
out=ROOT/'experiments/09162026/09162026_ieee9_moe_diagnosis'
out.mkdir(exist_ok=False)
(out/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
for t,r in report.items():
    print(t,json.dumps({k:v for k,v in r.items() if k not in ['validation_history','training_selected_counts_best']}),flush=True)
