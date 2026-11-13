"""Validate preparation and baseline adapters before the bounded pilot."""
import argparse,gc,json,pathlib,subprocess,sys,time
import numpy as np
import torch
from common import n
from pathway_policy import PathwayMoE
from run import distill,optimizer
from engine import phase

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--prepared-from',type=pathlib.Path);args=p.parse_args()
    out=args.output.expanduser();assert not out.exists();out.mkdir(parents=True)
    n.original.setup();lease=n.original.acquire();start=time.monotonic()
    try:
        state=n.load_parent();a=state['algorithm'];model=PathwayMoE(a.actor_net).cuda()
        if args.prepared_from:
            source=args.prepared_from.expanduser();results=json.loads((source/'DISTILLATION.json').read_text())
            model.load_state_dict(torch.load(source/'INITIAL_DISTILLED.pt',map_location='cuda',weights_only=True)['actor_state'])
        else:results=distill(a,model,out,steps=300)
        assert all(v['improved'] for v in results.values())
        # Exercise the standalone native path as well as selector validation.
        single=model.standalone(1);single.expert.requires_grad_(True);optimizer(a,single)
        value=phase(a,'standalone_native_validation',1,out,start,600,seed=202611134399)
        assert value['status']=='COMPLETE' and value['updates']==1
        del a,state,model,single;gc.collect();torch.cuda.empty_cache()
        cfg=dict(contrasts=31,contrast_seed=202611134203)
        (out/'CONFIG.json').write_text(json.dumps(cfg))
        rng=np.random.default_rng(202611134201)
        np.savez_compressed(out/'EVALUATION_CASES.npz',truth=rng.standard_normal((2,4)),noise=rng.standard_normal((2,30)))
        baselines={}
        for method in ['dad','rl_sboed','scorebed']:
            subprocess.run([sys.executable,'-u',str(pathlib.Path(__file__).parent/'baseline_worker.py'),'--method',method,'--output',str(out),'--smoke'],check=True)
            baselines[method]=json.loads((out/f'{method}_BASELINE.json').read_text())
        n.put(out/'VALIDATION.json',dict(status='PASS',head_distillation=results,standalone_native_updates=1,
             historical_baselines=baselines,seconds=time.monotonic()-start))
        print((out/'VALIDATION.json').read_text(),flush=True)
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
