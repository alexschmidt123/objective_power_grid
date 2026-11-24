"""Exact historical V2-bank comparison; frozen policies,no training."""
import argparse,gc,hashlib,io,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
V3SOURCE=WORK/'development/moe_sboed/hec_v3_v2_pathways_20261113';sys.path.insert(0,str(V3SOURCE))
from common import n,audit
from pathway_policy import PathwayMoE
from run import rollout
from resolution import nested_scores,paired
BASE=audit.BASE;PILOT=ROOT/'scratch_runs/hec_v3_v2_pathways_20261113_pilot'
UNIFORM_SAVED=ROOT/'scratch_runs/moe_hec_v2_internal_ablation/uniform_scores.npy'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required')
    out=args.output.expanduser();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    bank_path=BASE/'evaluation/development_validation_bank.npz';u_path=BASE/'evaluation/routing_uniforms.npy'
    actor=PILOT/'selector/actor.pt';receipt=json.loads((PILOT/'selector/RESULT.json').read_text());assert n.sha(actor)==receipt['actor_sha256']
    protected={str(p):n.sha(p) for p in [bank_path,u_path,BASE/'evaluation/candidate_scores.npy',UNIFORM_SAVED,
         actor,BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl',*sorted(V3SOURCE.glob('*.py')),*sorted(pathlib.Path(__file__).parent.glob('*.py'))]}
    n.put(out/'PRE_RUN_MANIFEST.json',protected);n.put(out/'CONFIG.json',dict(training_updates=0,bank='Exact historical V2 bank;not newly sampled',
         source_bank=str(bank_path),source_bank_sha256=protected[str(bank_path)],cases=2000,contrasts=1000000,
         routing_uniforms='Original saved V2 routing uniforms,also used for V3 learned/uniform categorical selection',
         reproducibility='Require full saved V2 learned/uniform score vectors reproduced within1e-9',wall_seconds=1800))
    n.original.setup()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+1800
    try:
        lib=n.load_library();v2=lib.load_frozen('cuda');v3=PathwayMoE(v2).cuda()
        v3.load_state_dict(torch.load(actor,map_location='cuda',weights_only=True)['actor_state']);v3.eval().requires_grad_(False)
        before2=n.digest(v2.state_dict());before3=n.digest(v3.state_dict())
        bank=np.load(bank_path);assert bank['truth'].shape==(2000,4) and bank['noise'].shape==(2000,30) and bank['contrasts'].shape==(1000000,4)
        theta=torch.tensor(bank['truth'],device='cuda');noise=torch.tensor(bank['noise'],device='cuda');neg=torch.tensor(bank['contrasts'],device='cuda')
        uniforms=torch.tensor(np.load(u_path),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
        scores={};counts={};reproduction={}
        for mode in ['V2','V2_uniform','V3','V3_uniform','expert1','expert2','expert3','expert4']:
            values=[];selected=torch.zeros(4,device='cuda',dtype=torch.long)
            for first in range(0,2000,16):
                if time.monotonic()>deadline:raise TimeoutError('Historical comparison cap')
                th,eps,u=theta[first:first+16],noise[first:first+16],uniforms[first:first+16]
                if mode in ['V2','V2_uniform']:
                    ds,ys,_=lib.rollout(v2,th,eps,'uniform' if mode=='V2_uniform' else 'original',deadline=deadline)
                else:
                    ds,ys,c=rollout(v3,lib,th,eps,u,'expert' if mode.startswith('expert') else 'uniform' if mode=='V3_uniform' else 'learned',int(mode[-1])-1 if mode.startswith('expert') else None)
                    selected+=c
                values.append(nested_scores(ds,ys,th,neg,[1000000],deadline=deadline)[1000000].cpu().numpy())
                n.put(out/'STATUS.json',dict(status='EVALUATING',method=mode,completed=min(first+16,2000),total=2000,seconds=time.monotonic()-start))
            scores[mode]=np.concatenate(values);counts[mode]=selected.cpu().tolist();np.save(out/f'{mode}_scores.npy',scores[mode])
            if mode in ['V2','V2_uniform']:
                saved=np.load(BASE/'evaluation/candidate_scores.npy') if mode=='V2' else np.load(UNIFORM_SAVED)
                error=float(np.abs(scores[mode]-saved).max());reproduction[mode]=dict(max_abs_score_error=error,mean=float(scores[mode].mean()),matched=error<1e-9)
                assert error<1e-9,f'{mode} historical scores not reproduced: {error}'
            n.put(out/'PARTIAL.json',dict(means={k:float(v.mean()) for k,v in scores.items()},reproduction=reproduction))
            print('EVALUATED',mode,float(scores[mode].mean()),flush=True)
        result=dict(status='COMPLETE',cases=2000,contrasts=1000000,means={k:float(v.mean()) for k,v in scores.items()},
            primary_V3_minus_V2=paired(scores['V3'],scores['V2']),V3_minus_uniform=paired(scores['V3'],scores['V3_uniform']),
            V2_reproduction=reproduction,single_comparisons={f'E{k}':paired(scores['V3'],scores[f'expert{k}']) for k in range(1,5)},
            counts=counts,seconds=time.monotonic()-start,
            fairness='Identical original V2 physical experiment,truth/noise cases,prior contrast bank,likelihood,bounds and horizon. Evaluation matched;V3 has additional training and compute,not training-budget matched to frozen V2.',
            limitations='Historical1M prior-sPCE is ceiling-limited;use16M paired results for better-resolved conclusions. Frozen checkpoint comparison,one lineage,no training.')
        assert n.digest(v2.state_dict())==before2 and n.digest(v3.state_dict())==before3
        assert all(n.sha(p)==h for p,h in protected.items())
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',historical_artifacts_preserved=True,seconds=time.monotonic()-start,training_updates=0))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
