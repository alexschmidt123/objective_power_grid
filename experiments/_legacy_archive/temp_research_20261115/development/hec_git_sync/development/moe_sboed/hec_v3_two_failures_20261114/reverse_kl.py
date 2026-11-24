"""Correct analytic KL optimum; preserve the original measured-gradient artifact."""
import json
from diagnostic_common import ROOT,CF,np,torch,n,module

@torch.no_grad()
def optimum(old,advantages):
    # -sum(p*A)+KL(old||p): KKT gives p=old/(tau-A), not old*exp(A).
    p=old.double();a=advantages.double();low=a.max(-1,keepdim=True).values+1e-12;high=low+1
    for _ in range(80):
        mid=(low+high)/2;mass=(p/(mid-a)).sum(-1,keepdim=True)
        low=torch.where(mass>1,mid,low);high=torch.where(mass>1,high,mid)
    tau=(low+high)/2;q=p/(tau-a)
    torch.testing.assert_close(q.sum(-1),torch.ones(len(q),device=q.device,dtype=q.dtype),atol=1e-10,rtol=1e-10)
    assert (q>0).all() and (-a-p/q+tau).abs().max()<1e-10
    return q

def main():
    out=ROOT/'scratch_runs/hec_v3_two_failures_20261114';original=json.loads((out/'RESULTS.json').read_text())
    p=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64);a=torch.zeros_like(p)
    torch.testing.assert_close(optimum(p,a),p,atol=1e-10,rtol=1e-10)
    a=torch.tensor([[.2,-.3,2.,.1]],dtype=torch.float64);q=optimum(p,a)
    def objective(v):return -(v*a).sum()+(p*(p.log()-v.log())).sum()
    assert objective(q)<=objective(p)
    n.original.setup();lease=n.original.acquire()
    try:
        lib=n.load_library();parent=lib.load_frozen('cuda');model=module.d.PathwayMoE(parent).cuda()
        previous=module.PILOT/'selector/actor.pt'
        for r,row in enumerate(original['frozen_tests'],1):
            data=np.load(CF/f'counterfactual_round{r}/DATA.npz');ck=torch.load(CF/f'counterfactual_round{r}/value_ensemble.pt',map_location='cuda',weights_only=True)
            model.load_state_dict(torch.load(previous,map_location='cuda',weights_only=True)['actor_state']);model.eval().requires_grad_(False)
            with torch.no_grad():
                x=torch.tensor(data['history'][:64],device='cuda');old=model.components(x)[1]
                pred=module.ensemble(ck,torch.tensor(data['features'],device='cuda'))[:64]
                q=optimum(old,pred);tv=float((q-old.double()).abs().sum(-1).mean()/2)
            row['router_update'].pop('mean_unrestricted_KL_penalized_optimum_TV')
            row['router_update']['mean_reverse_KL_optimum_TV']=tv
            row['router_update']['explanation']='For KL(old||new), p*=old/(tau-A), with tau chosen to normalize. Exponential reweighting belongs to KL(new||old) and does not apply here.'
            previous=CF/f'counterfactual_round{r}/actor.pt'
        original['analytic_correction']='Only the analytic optimum calculation in the original frozen RESULTS.json was corrected. Measured gradients, actual changes, bridge scores and historical files remain unchanged. Original raw file is preserved.'
        (out/'CORRECTED_RESULTS.json').write_text(json.dumps(original,indent=2)+'\n')
        print(json.dumps(dict(status='PASS',zero_advantage_identity=True,KKT_normalization=True,objective_decrease=True,
            corrected_TV=[row['router_update']['mean_reverse_KL_optimum_TV'] for row in original['frozen_tests']])) )
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
