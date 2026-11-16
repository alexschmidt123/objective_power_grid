"""Read-only heldout value diagnostics; never used to choose training settings."""
import argparse,json,pathlib
import numpy as np
import torch
from torch import nn
from experiment import ROOT,WORK,advantage
from resolution import paired
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();out=args.output.expanduser()
assert ROOT/'scratch_runs' in out.parents
results=[]
for path in sorted(out.glob('*_round*')):
    bank=np.load(path/'DATA.npz');ck=torch.load(path/'value_ensemble.pt',map_location='cpu',weights_only=True)
    x=torch.tensor(bank['features']);z=((x-ck['mean'])/ck['std']).clamp(-20,20);pred=[]
    with torch.no_grad():
        for state in ck['models']:
            net=nn.Sequential(nn.Linear(648,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,4));net.load_state_dict(state)
            pred.append(advantage(net(z))*ck['scale'])
    prediction=torch.stack(pred).mean(0).numpy();sl=slice(80,96)
    test=bank['test'][sl];mean=test.mean(1);actual=mean-mean.mean(-1,keepdims=True)
    centered=test-test.mean(-1,keepdims=True)
    noise=float(centered.var(axis=1,ddof=1).mean()/test.shape[1])
    naive=bank['selection'][sl].mean(1).argmax(-1);rows=np.arange(16)
    quality=json.loads((path/'POSTERIOR_QUALITY.json').read_text());approx=[q for q in quality if not q.get('exact_prior')]
    results.append(dict(round=path.name,audit_histories=16,
        ensemble_advantage_MSE=float(np.square(prediction[sl]-actual).mean()),zero_advantage_MSE=float(np.square(actual).mean()),
        critic_advantage_MSE=float(np.square(bank['critic'][sl]-bank['critic'][sl].mean(-1,keepdims=True)-actual).mean()),
        estimated_suffix_mean_noise_MSE=noise,
        empirical_selection_minus_E3=paired(mean[rows,naive],mean[:,2]),
        empirical_selection_minus_current=paired(mean[rows,naive],(mean*bank['weights'][sl]).sum(-1)),
        minimum_ESS=min(q['ESS'] for q in approx),maximum_weight=max(q['max_weight'] for q in approx),
        largest_particle_budget=max(q['particles'] for q in approx)))
dest=WORK/'reports/hec_v3_counterfactual_router_20261114';dest.mkdir(parents=True,exist_ok=True)
result=dict(rounds=results,note='Posthoc descriptive checks on untouched audit histories; no tuning. Independent suffix draws share finite importance-posterior approximation. Empirical selection is noisy and is not an oracle upper bound.')
(dest/'VALUE_DIAGNOSTICS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
