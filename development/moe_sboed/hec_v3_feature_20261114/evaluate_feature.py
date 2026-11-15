"""Exact original V2 evaluation for replacement V3; frozen reference certificates."""
import copy,json,pathlib,sys,time
import numpy as np
import torch
from v3_native import n,ROOT,WORK
from feature_policy import FeatureMoE
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_canonical_evaluation_20261114'))
from protocol import load,LOCK,sha
from resolution import signal,paired
CANON=ROOT/'scratch_runs/hec_canonical_evaluation_20261114'
def references():
    certificate=json.loads((WORK/'reports/hec_canonical_evaluation_20261114/VERIFICATION.json').read_text())
    assert certificate['protocol']==LOCK and all(v['max_abs_error']==0 for v in certificate['historical_reproduction'].values())
    inventory=json.loads((n.BASE/'inputs/HISTORICAL_BASELINES.json').read_text())
    paths=dict(V2=n.BASE/'arms/candidate/actor.pt',vsoed=n.BASE/'inputs/vsoed.pt',
        original_V3=ROOT/'scratch_runs/hec_v3_v2_pathways_20261113_pilot/selector/actor.pt',
        counterfactual=ROOT/'scratch_runs/hec_v3_counterfactual_router_20261114_pilot02/counterfactual_round3/actor.pt',
        critic_control=ROOT/'scratch_runs/hec_v3_counterfactual_router_20261114_pilot02/critic_control_round3/actor.pt')
    paths.update({r['method']:pathlib.Path(r['checkpoint']) for r in inventory['rows']})
    assert all(sha(p)==certificate['checkpoint_sha256'][k] for k,p in paths.items())
    scores={};receipts={}
    for key in ['V2','V2_uniform','vsoed','dad','rl_sboed','scorebed']:
        source=n.BASE/'evaluation'/('candidate_scores.npy' if key=='V2' else key+'_scores.npy')
        if key=='V2_uniform':source=ROOT/'scratch_runs/moe_hec_v2_internal_ablation/uniform_scores.npy'
        expected=certificate['historical_reproduction'][key]['saved_scores_sha256'];assert sha(source)==expected
        value=np.load(CANON/f'{key}_scores.npy');assert np.array_equal(value,np.load(source))
        scores[key]=value;receipts[key]=dict(status='VERIFIED_EXACT_CACHED_REPRODUCTION',score_sha256=sha(source),max_abs_error=0.)
    for key in ['original_V3','counterfactual']:
        value=np.load(CANON/f'{key}_scores.npy');scores['retired_'+key]=value
    return scores,receipts

@torch.no_grad()
def evaluate(out,deadline):
    bank,u,score=load();lib=n.load_library();parent=lib.load_frozen('cuda');model=FeatureMoE(parent).cuda()
    model.load_state_dict(torch.load(out/'candidate_joint/actor.pt',map_location='cuda',weights_only=True)['actor_state']);model.configure('frozen');model.eval()
    control=copy.deepcopy(parent);control.load_state_dict(torch.load(out/'control/actor.pt',map_location='cuda',weights_only=True)['actor_state'])
    scores,certificates=references();neg=torch.tensor(bank['contrasts'],device='cuda');counts={}
    for mode in ['learned','uniform','expert1','expert2','expert3','expert4','control','V2_reproduction']:
        values=[]
        for i in range(0,2000,16):
            if time.monotonic()>deadline:raise TimeoutError('Evaluation cap')
            th=torch.tensor(bank['truth'][i:i+16],device='cuda');noise=torch.tensor(bank['noise'][i:i+16],device='cuda')
            ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,device='cuda',dtype=torch.float64)
            for t in range(30):
                x=lib.packed(ds,ys,t)
                if mode in ['control','V2_reproduction']:action=(control if mode=='control' else parent)(x)
                else:action=model(x,'expert' if mode.startswith('expert') else mode,int(mode[-1])-1 if mode.startswith('expert') else None)
                assert torch.isfinite(action).all() and (action.abs()<=4.000001).all()
                ds[:,t]=action;ys[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*noise[:,t]
            result,_=score(type('Benchmark',(),{'sigma':.5,'calls':0})(),dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th),neg,16384,'cuda')
            values.extend(result[:,-1].numpy());n.put(out/'STATUS.json',dict(status='EVALUATING',method=mode,completed=min(i+16,2000),total=2000))
        scores[mode]=np.array(values);np.save(out/f'{mode}_scores.npy',scores[mode])
    assert np.abs(scores['V2_reproduction']-scores['V2']).max()<1e-9
    comparisons={k:paired(scores['learned'],v) for k,v in scores.items() if k not in ['learned','V2_reproduction']};largest=0
    for j,k in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['p_two_sided'])):
        largest=max(largest,min(1,(len(comparisons)-j)*comparisons[k]['p_two_sided']));comparisons[k]['p_Holm']=largest
    accepted=all(comparisons[k]['mean']>0 and comparisons[k]['p_Holm']<.05 for k in ['uniform','expert1','expert2','expert3','expert4','V2','control','dad','vsoed','rl_sboed','scorebed'])
    result=dict(status='COMPLETE',protocol=LOCK,means={k:float(v.mean()) for k,v in scores.items()},comparisons=comparisons,
        accepted=accepted,baseline_certificates=certificates,reference_reuse='Byte-verified exact saved reference vectors on unchanged bank/scorer/checkpoints;V2 additionally rerolled and rescored.',
        limitations='Short one-lineage pilot,internal standalone ablations not independently optimized baselines;native-update/simulator budgets matched to control,parameter counts and FLOPs differ;1M prior-sPCE ceiling limited.')
    n.put(out/'RESULTS.json',result);return result
