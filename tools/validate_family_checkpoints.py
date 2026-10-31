"""Read-only compatibility replay and measured parameter counts; never trains."""
if __name__ == '__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: historical result generation is archived')
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.moe_family import load_policy,MoESBOEDPolicy,BasePolicy,measured_parameters


def validate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    old=root/'experiments/10142026/10142026_223452_sum_moe_slower_sharpening/pilot'
    switch=root/'experiments/10142026/10142026_232132_strategy_moe_pilot/pilot'
    dense=root/'experiments/10142026/10142026_moe_controlled_diagnosis/controls'
    report={};counts={}
    for variant,run,stem in [('moe',old,'moe_sboed'),('switchmoe',switch,'strategy_moe_sboed')]:
        path=run/'models'/(stem+'.pt');before=hashlib.sha256(path.read_bytes()).hexdigest()
        raw=json.loads((run/'run_config.json').read_text())['benchmark'];b=make_benchmark(raw)
        rows=json.loads((run/'eval'/(stem+'_seed1001.json')).read_text())[:32]
        p=load_policy(path,b);truth=torch.tensor([r['truth'] for r in rows],dtype=torch.float64)
        noise=torch.tensor([r['measurement_noise_standard'] for r in rows],dtype=torch.float64)
        with torch.no_grad():r=rollout(b,p,30,len(rows),1001,0,truth=truth,noise=noise)
        a=torch.stack(r['actions'],1);y=torch.stack(r['observations'],1)
        expected=torch.tensor([v['actions'] for v in rows],dtype=torch.float64)
        obs=torch.tensor([v['observations'] for v in rows],dtype=torch.float64)
        torch.testing.assert_close(a,expected,atol=1e-10,rtol=0)
        torch.testing.assert_close(y,obs,atol=1e-10,rtol=0)
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        report[variant]=dict(checkpoint=str(path),sha256=before,paired_trajectories=len(rows),
            max_action_error=float((a-expected).abs().max()),max_observation_error=float((y-obs).abs().max()),
            tolerance=1e-10,historical_eig=np.mean([v['spce'] for v in json.loads((run/'eval'/(stem+'_seed1001.json')).read_text())]),
            eig_recomputed=False,note='Identical histories/actions under paired randomness preserve the historical result; no new performance evaluation')
        counts[variant]=dict(first=measured_parameters(p,r['features'][0][:1],0),initial_routed=measured_parameters(p,r['features'][1][:1],1),later_routed=measured_parameters(p,r['features'][2][:1],2))
    b=make_benchmark(json.loads((dense/'run_config.json').read_text())['benchmark'])
    bp=dense/'models/sum_single.pt';before=hashlib.sha256(bp.read_bytes()).hexdigest()
    for v in ['resmoe','switchresmoe']:
        p=MoESBOEDPolicy(v,b,30,base_checkpoint=bp,residual_scale=0.).eval()
        with torch.no_grad():
            a=rollout(b,p,30,32,1001,8);c=rollout(b,p.core.base.policy,30,32,1001,8)
        for key in ['actions','observations','latents']:torch.testing.assert_close(torch.stack(a[key]),torch.stack(c[key]),atol=0,rtol=0)
        torch.testing.assert_close(a['info'],c['info'],atol=0,rtol=0)
        report[v]=dict(base_sha256=before,base_disabled_exact=True,paired_trajectories=32,tolerance=0.)
        p.core.alpha=1.
        p.set_routing()
        counts[v]=dict(first=measured_parameters(p,a['features'][0][:1],0),initial_routed=measured_parameters(p,a['features'][1][:1],1),later_routed=measured_parameters(p,a['features'][2][:1],2))
    assert before==hashlib.sha256(bp.read_bytes()).hexdigest()
    (output/'checkpoint_compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'measured_parameters.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(dict(compatibility=report,parameters=counts),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(1)
    validate(Path(__file__).resolve().parents[1],args.output)
