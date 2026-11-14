"""Primary matched evaluation of frozen routers; no training, original V2 scorer."""
import argparse,copy,json,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from protocol import ROOT,WORK,BASE,LOCK,BANK,UNIFORMS,SCORER,sha,load,tests
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_v3_counterfactual_router_20261114'))
from experiment import n,d,PILOT,rollout,paired
CF=ROOT/'scratch_runs/hec_v3_counterfactual_router_20261114_pilot02'
UNIFORM_SAVED=ROOT/'scratch_runs/moe_hec_v2_internal_ablation/uniform_scores.npy'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    out=args.output.expanduser();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    n.put(out/'GUARD_TESTS.json',tests());bank,uniforms,score=load()
    artifacts=json.loads((WORK/'reports/hec_v3_counterfactual_router_20261114/ARTIFACTS.json').read_text())
    actors={name:CF/f'{folder}_round3/actor.pt' for name,folder in [('counterfactual','counterfactual'),('critic_control','critic_control')]}
    for name,path in actors.items():assert sha(path)==artifacts['new_checkpoint_hashes'][str(path.relative_to(CF))]
    inventory=json.loads((BASE/'inputs/HISTORICAL_BASELINES.json').read_text())
    checkpoints=dict(actors,V2=BASE/'arms/candidate/actor.pt',original_V3=PILOT/'selector/actor.pt',vsoed=BASE/'inputs/vsoed.pt')
    for row in inventory['rows']:
        checkpoints[row['method']]=pathlib.Path(row['checkpoint']);assert sha(row['checkpoint'])==row['checkpoint_sha256']
    assert sha(checkpoints['V2'])=='90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108'
    assert sha(checkpoints['original_V3'])==json.loads((PILOT/'selector/RESULT.json').read_text())['actor_sha256']
    checkpoint_hashes={k:sha(v) for k,v in checkpoints.items()};n.put(out/'CHECKPOINTS.json',checkpoint_hashes)
    protected=[BANK,UNIFORMS,SCORER,UNIFORM_SAVED,PILOT/'selector/actor.pt',*actors.values(),BASE/'inputs/vsoed.pt',
        *sorted((BASE/'evaluation').glob('*_scores.npy')),*sorted(pathlib.Path(__file__).parent.glob('*.py'))]
    protected+=[BASE/'inputs/HISTORICAL_BASELINES.json',*checkpoints.values(),*[pathlib.Path(row['arrays_path']) for row in inventory['rows'] if row.get('arrays_path')]]
    manifest={str(f):sha(f) for f in protected};n.put(out/'MANIFEST.json',manifest)
    n.put(out/'CONFIG.json',dict(**LOCK,training_updates=0,bank=str(BANK),scorer=str(SCORER),
        source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip(),
        training_budget_matching=False,primary_protocol='EXACT_ORIGINAL_V2_DAD_VSOED',positive_truth_included_once=True))
    n.original.setup();lease=None
    while lease is None:
        try:lease=n.original.acquire()
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+1800
    try:
        lib=n.load_library();v2=lib.load_frozen('cuda');original=d.PathwayMoE(v2).cuda()
        original.load_state_dict(torch.load(PILOT/'selector/actor.pt',map_location='cuda',weights_only=True)['actor_state'])
        original.eval().requires_grad_(False);models={'original_V3':original}
        for name,path in actors.items():
            model=copy.deepcopy(original);model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['actor_state']);models[name]=model.eval().requires_grad_(False)
        import evaluate_v2
        vsoed,_=evaluate_v2.actor_load('vsoed');models['vsoed']=vsoed
        expected_vsoed=json.loads((BASE/'ARTIFACT_SHA256.json').read_text())['files']['inputs/vsoed.pt'];assert sha(BASE/'inputs/vsoed.pt')==expected_vsoed
        states={k:n.digest(m.state_dict()) for k,m in dict(models,V2=v2).items()}
        th=torch.tensor(bank['truth'],device='cuda');eps=torch.tensor(bank['noise'],device='cuda');neg=torch.tensor(bank['contrasts'],device='cuda')
        u=torch.tensor(uniforms,device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
        scores={};repro={};counts={}
        modes=['V2','V2_uniform','vsoed','original_V3','counterfactual','critic_control','uniform','expert1','expert2','expert3','expert4']
        for name in modes:
            vals=[];curves=[];selected=torch.zeros(4,device='cuda',dtype=torch.long)
            for i in range(0,2000,16):
                if time.monotonic()>deadline:raise TimeoutError('Primary evaluation wall cap')
                with torch.no_grad(),torch.device('cuda'):
                    if name.startswith('V2'):
                        ds,ys,_=lib.rollout(v2,th[i:i+16],eps[i:i+16],'uniform' if name=='V2_uniform' else 'original',deadline=deadline)
                    else:
                        model=models.get(name,original)
                        ds,ys,c=rollout(model,lib,th[i:i+16],eps[i:i+16],u[i:i+16],
                            'expert' if name.startswith('expert') else 'uniform' if name=='uniform' else 'learned',int(name[-1])-1 if name.startswith('expert') else None);selected+=c
                    data=dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th[i:i+16])
                    result,_=score(type('Benchmark',(),{'sigma':.5,'calls':0})(),data,neg,16384,'cuda')
                assert torch.isfinite(result).all();vals.extend(result[:,-1].numpy());curves.append(result.numpy())
                n.put(out/'STATUS.json',dict(status='EVALUATING',method=name,completed=min(i+16,2000),total=2000,seconds=time.monotonic()-start))
            scores[name]=np.array(vals);counts[name]=selected.cpu().tolist();np.save(out/f'{name}_scores.npy',scores[name]);np.save(out/f'{name}_curves.npy',np.concatenate(curves))
            reference=BASE/'evaluation/candidate_scores.npy' if name=='V2' else UNIFORM_SAVED if name=='V2_uniform' else BASE/'evaluation/vsoed_scores.npy' if name=='vsoed' else None
            if reference:
                error=float(np.abs(scores[name]-np.load(reference)).max());assert error<1e-9,f'{name} reproduction failed {error}'
                repro[name]=dict(max_abs_error=error,saved_scores_sha256=sha(reference))
        for name in ['dad','rl_sboed','scorebed']:
            n.put(out/'STATUS.json',dict(status='EVALUATING',method=name))
            subprocess.run([sys.executable,str(pathlib.Path(__file__).parent/'baseline_worker.py'),'--method',name,'--output',str(out)],check=True,timeout=max(1,deadline-time.monotonic()))
            scores[name]=np.load(out/f'{name}_scores.npy');repro[name]=json.loads((out/f'{name}_REPRODUCTION.json').read_text())
        comparisons={name:paired(scores['counterfactual'],s) for name,s in scores.items() if name!='counterfactual'}
        maximum=0
        for j,name in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['p_two_sided'])):
            maximum=max(maximum,min(1,(len(comparisons)-j)*comparisons[name]['p_two_sided']));comparisons[name]['p_Holm']=maximum
        assert all(sha(path)==h for path,h in manifest.items())
        assert all(n.digest(m.state_dict())==states[k] for k,m in dict(models,V2=v2).items())
        result=dict(status='COMPLETE',protocol=LOCK,means={k:float(s.mean()) for k,s in scores.items()},comparisons=comparisons,
            historical_reproduction=repro,selection_counts=counts,seconds=time.monotonic()-start,training_updates=0,
            historical_artifacts_preserved=True,checkpoint_sha256=checkpoint_hashes,limitations='Exact original evaluation matched; training budgets differ. Original1M prior-sPCE is ceiling limited;one lineage,not official promoted results.16M fresh-bank run is supplementary only.')
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',seconds=result['seconds'],training_updates=0))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
