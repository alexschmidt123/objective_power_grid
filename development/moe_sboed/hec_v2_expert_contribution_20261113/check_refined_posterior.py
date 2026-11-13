"""No-EIG scale check of the fixed refined posterior on failed history."""
import json,pathlib,time,os,sys
import numpy as np
import torch
import diagnostic as d
def main():
    os.environ['MOE_V2_RUN_ROOT']=str(d.BASE);sys.path.insert(0,str(d.BASE/'source'))
    from native_common import original
    original.setup();lease=original.acquire()
    try:
        lib=d.old.load_library();actor=lib.load_frozen('cuda')
        run=d.ROOT/'scratch_runs/hec_v2_expert_contribution_20261113'
        data=np.load(run/'test_HISTORIES.npz');stage=int(data['stages'][23])
        ds=torch.tensor(data['actions'][23:24],device='cuda')
        ys=torch.tensor(data['observations'][23:24],device='cuda')
        x=lib.packed(ds,ys,stage);start=time.monotonic()
        _,quality=d.old.posterior(actor,x,ds,ys,stage,64,2097152,202611241924)
        result=dict(status='PASS' if quality['ESS']>=256 and quality['max_weight']<=.05 else 'INCONCLUSIVE',
                    seconds=time.monotonic()-start,quality=quality,EIG_evaluations=0,optimizer_steps=0)
        path=d.ROOT/'scratch_runs/hec_v2_expert_contribution_20261113_refined_scale_check.json'
        assert not path.exists();d.put(path,result);print(json.dumps(result),flush=True)
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
