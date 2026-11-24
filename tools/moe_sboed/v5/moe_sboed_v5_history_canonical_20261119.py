"""Primary canonical evaluation of replacement V5 terminal checkpoints;zero training."""
from pathlib import Path
import importlib.util,json,sys,time,subprocess,fcntl
import numpy as np
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v5_history_canonical_20261119';R=M/'reports/moe_sboed_v5_history_canonical_20261119'
spec=importlib.util.spec_from_file_location('canonical_history_v5',M/'src/research/moe_sboed_v5_history_20261119/run.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
import torch

def read(path):return json.loads(Path(path).read_text())
def put(path,data):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(path,data)

def main():
 assert not O.exists() and not R.exists(),'Preserve existing canonical evaluation;no overwrite'
 O.mkdir(parents=True);R.mkdir(parents=True);lock=(O/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  cert=p.engine.verify_references();bank,uniforms,scorer=p.engine.canonical_load();lockid=p.engine.LOCK
  protocol=dict(status='LOCKED',evaluation=lockid,training_seed=101,training_updates=0,checkpoint_selection='Existing terminal2000 checkpoints,no selection',primary='Original canonical V2/V4/DAD/vsOED bank and scorer;new-bank V5 pilot remains supplementary',historical_verification=cert['historical_reproduction'],source_sha256=p.sha(Path(__file__)))
  put(R/'PROTOCOL.json',protocol)
  records=read(M/'reports/moe_sboed_v4_allmethods_20261117/RESULTS.json')['records'];saved={r['method']:r for r in records if r['seed']==101}
  for key in ['moe-sboed','moe-sboed-uniform','vsOED','DAD']:
   row=saved[key];assert p.sha(row['checkpoint'])==row['checkpoint_sha256'];assert p.sha(row['scores'])==row['scores_sha256']
  p.setup();rows={};vectors={};reproduction={};start=time.monotonic()
  with p.lease(),p.engine.Resources(O):
   state=p.load_parent('cuda');parent=state['algorithm'].actor_net
   jobs=[('previous_learned','Original V4 learned','moe-sboed'),('previous_uniform','Original V4-uniform','moe-sboed-uniform'),('vsoed','vsOED','vsOED'),('learned','MoE-sBOED V5',None),('uniform','MoE-sBOED V5-uniform',None)]
   for key,label,control in jobs:
    put(O/'STATUS.json',dict(status='EVALUATING',current=label,completed=list(rows),pending=[x[1] for x in jobs[len(rows)+1:]]))
    folder=O/key;folder.mkdir()
    if control:actor,cp=p.reference_actor(key,parent)
    else:
     actor=p.new_actor(parent,key);cp=p.O/key/'joint_actor.pt';actor.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['actor_state']);actor.configure('frozen');actor.eval()
    before=p.sha(cp);actor.eval()
    with torch.no_grad():values,ev=p.engine.evaluate(actor,folder,key,time.monotonic()+1800,collect=False)
    assert p.sha(cp)==before and ev['protocol']==lockid
    if control:
     assert before==saved[control]['checkpoint_sha256'];difference=float(np.max(np.abs(values-np.load(saved[control]['scores']))));reproduction[control]=dict(max_abs_score_error=difference,checkpoint_sha256=before,saved_scores_sha256=saved[control]['scores_sha256']);assert difference<=1e-9,(control,difference)
    rows[key]=dict(method=label,seed=101,**p.engine.summaries(values),checkpoint=str(cp),checkpoint_sha256=before,evaluation=ev,scores=str(folder/(key+'_scores.npy')));vectors[key]=values
    put(folder/'RESULTS.json',rows[key]);del actor;torch.cuda.empty_cache()
   row=saved['DAD'];values=np.load(row['scores']);rows['dad']=dict(method='DAD',seed=101,**p.engine.summaries(values),checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],scores=row['scores'],reused_verified_saved_canonical_scores=True)
   pairs=dict(V5_minus_V4=vectors['learned']-vectors['previous_learned'],learned_minus_uniform=vectors['learned']-vectors['uniform'],V5_minus_vsOED=vectors['learned']-vectors['vsoed'],uniform_V5_minus_V4=vectors['uniform']-vectors['previous_uniform'],V5_minus_DAD=vectors['learned']-values,
    gap_improvement=(vectors['learned']-vectors['uniform'])-(vectors['previous_learned']-vectors['previous_uniform']))
   comparisons={key:p.engine.summaries(value) for key,value in pairs.items()}
   result=dict(status='COMPLETE',primary_protocol=lockid,rows=rows,comparisons=comparisons,controls_reproduced=reproduction,case_uncertainty_only=True,training_seed_count=1,training_updates=0,seconds=time.monotonic()-start,
    conclusion='Report primary canonical results separately from supplementary new-bank development pilot. Do not compare seed101 against five-seed means. Finite1M-contrast EIG lower bounds;MCSE excludes contrast bias.')
   put(R/'RESULTS.json',result);put(R/'CONTROL_REPRODUCTION.json',reproduction)
   text='# Replacement V5: PRIMARY canonical evaluation,seed101\n\nOriginal V2/V4/DAD/vsOED bank;2000 trajectories,1M contrasts,T30. Supplementary development-bank results remain unchanged.\n\n|Method|EIG lower bound (nats)|MCSE|\n|---|---:|---:|\n'
   for key in ['learned','uniform','previous_learned','previous_uniform','vsoed','dad']:
    r=rows[key];text+=f"|{r['method']}|{r['mean']:.5f}|{r['MCSE']:.5f}|\n"
   text+='\n|Paired comparison|Difference|95% case evaluation CI|\n|---|---:|---|\n'
   for key,r in comparisons.items():text+=f"|{key}|{r['mean']:+.5f}|[{r['CI95'][0]:+.5f},{r['CI95'][1]:+.5f}]|\n"
   text+='\nFresh V4 learned/uniform and vsOED score vectors reproduce saved canonical controls within1e-9. DAD canonical scores reused after saved-score/checkpoint hash validation. Historical V2/DAD/vsOED guard certificate and bank/scorer hashes validated. No training/checkpoint selection. One continuation seed,not five-seed robustness.\n';(R/'RESULTS.md').write_text(text)
  put(O/'STATUS.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
 except Exception as e:put(O/'STATUS.json',dict(status='FAILED_PRESERVED',error=repr(e)));raise
if __name__=='__main__':main()
