"""Repair canonical-bank metadata alias;recover learned evaluation,then run pending uniform only."""
from pathlib import Path
import fcntl,importlib.util,json,os,shutil,subprocess,sys,time
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v5_critic_20261120';R=M/'reports/moe_sboed_v5_critic_20261120'
spec=importlib.util.spec_from_file_location('v5_critic_recovery',M/'src/research/moe_sboed_v5_critic_20261120/run.py');s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
p=s.p
import numpy as np,torch

def read(path):return json.loads(Path(path).read_text())
def put(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(path,value)
def repair():
 assert s.source_identity()==read(O/'SOURCE_MANIFEST.json')
 source=p.n.BASE/'evaluation/development_validation_bank.npz';assert p.sha(source)==p.engine.LOCK['bank_sha256'];p.engine.verify_references()
 alias=O/'development_validation_bank.npz'
 if alias.is_symlink() or alias.exists():assert p.sha(alias)==p.engine.LOCK['bank_sha256']
 else:alias.symlink_to(source)
 put(R/'CANONICAL_BANK_PATH_REPAIR.json',dict(status='VERIFIED',alias=str(alias),source=str(source),sha256=p.sha(alias),new_bank_generated=False,policy_or_training_source_modified=False,
 reason='Original canonical loader was used correctly;legacy evaluator metadata expected a local bank filename. Alias supplies the identical frozen canonical file.'))

def evaluation():
 repair();out=O/'learned';assert not (out/'RESULTS.json').exists();start=time.monotonic();old_status=read(out/'STATUS_before_recovery.json');before_seconds=old_status['seconds'];credit=s.cfg()['preparation_credit_seconds']
 phases=[read(out/(name+'_TRAINING.json')) for name in ['warmup','joint']];assert all(x['status']=='COMPLETE' for x in phases)
 assert {k:sum(x['counts'][k] for x in phases) for k in ['actor','critic','posterior']}==dict(actor=2000,critic=10000,posterior=300000)
 p.setup()
 with p.lease(),p.engine.Resources(out/'evaluation_recovery_resources'):
  state=p.load_parent('cuda');model=s.ORIGINAL_NEW_ACTOR(state['algorithm'].actor_net,'learned');cp=out/'joint_actor.pt';before=p.sha(cp);checkpoint=torch.load(cp,map_location='cuda',weights_only=True);assert checkpoint['native_iteration']==18001
  model.load_state_dict(checkpoint['actor_state']);model.configure('frozen');model.eval();s.bind()
  # Keep the already computed score arrays and failed metadata attempt intact.
  old=out/'evaluation_attempt01_metadata_failure';original=np.load(old/'scores.npy')
  with torch.no_grad():ev=p.evaluate(model,out/'evaluation',s.NAMES['learned'],start+min(1200,14300-before_seconds-credit))
  values=np.load(out/'evaluation/scores.npy');error=float(np.max(np.abs(values-original)));assert error<=1e-9 and p.sha(cp)==before
  logs=[json.loads(line) for line in (out/'TRAINING_METRICS.jsonl').read_text().splitlines()];keys=set().union(*(r['gradient_group_norms'] for r in logs));groups={k:[r['gradient_group_norms'][k] for r in logs if k in r['gradient_group_norms']] for k in keys}
  grads={k:dict(mean=float(np.mean(v)),max=float(max(v)),min=float(min(v)),nonzero=sum(x>0 for x in v),snapshots=len(v)) for k,v in groups.items()}
  recovery_seconds=time.monotonic()-start;total=before_seconds+recovery_seconds;assert total+credit<=14400
  put(out/'RESULTS.json',dict(status='COMPLETE',method=s.NAMES['learned'],seed=101,updates=2000,phases=phases,training_seconds=sum(x['seconds'] for x in phases),total_seconds=total,
   new_training_episodes=2000000,new_training_observations=60000000,parameter_counts={},training_gradient_norms=grads,training_model_derived_calls=None,
   evaluation=ev,final_actor_sha256=before,historical_artifacts_preserved=True,recovery=dict(no_retraining=True,original_failed_worker_seconds=before_seconds,recovery_seconds=recovery_seconds,
   score_vector_max_abs_error=error,extra_evaluation_episodes=2000,extra_evaluation_observations=60000,training_timer_scope='Sum of preserved native phase timers;original encompassing stopwatch unavailable',model_query_counts='Unavailable after original worker exception;not imputed as zero')))
  put(out/'STATUS.json',dict(status='COMPLETE',completed_updates=2000,evaluation_metadata_recovered=True,no_retraining=True))
 put(R/'LEARNED_RECOVERY_RECEIPT.json',read(out/'RESULTS.json')['recovery']);s.report();print('LEARNED_EVALUATION_RECOVERED',flush=True)

def controller():
 handle=(O/'.recovery_controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'RECOVERY_LAUNCH.json').exists(),'Never duplicate a recovery attempt'
 assert read(O/'STATUS.json')['status']=='FAILED_PRESERVED' and not (O/'uniform').exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 backup=R/'pre_recovery_failure';backup.mkdir(exist_ok=False)
 for f in R.iterdir():
  if f.is_file() and f.suffix in ['.md','.json']:shutil.copy2(f,backup/f.name)
 shutil.copy2(O/'STATUS.json',O/'STATUS_before_recovery.json');shutil.copy2(O/'learned/STATUS.json',O/'learned/STATUS_before_recovery.json')
 shutil.copy2(O/'PUBLICATION_STATUS.json',O/'PUBLICATION_STATUS_before_recovery.json')
 (O/'learned/evaluation').rename(O/'learned/evaluation_attempt01_metadata_failure')
 (O/'learned/evaluation_recovery_resources').mkdir()
 put(O/'RECOVERY_LAUNCH.json',dict(pid=os.getpid(),reason='Evaluation metadata path failure after complete native training',no_model_or_objective_change=True))
 repair()
 jobs=[('recover_learned_evaluation',[sys.executable,'-B','-u',str(Path(__file__)),'--evaluate'],1200),('uniform',[sys.executable,'-B','-u',str(M/'tools/moe_sboed_v5_critic_20261120.py'),'--task','train','--arm','uniform'],int(14370-s.cfg()['preparation_credit_seconds']))]
 for name,command,seconds in jobs:
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(20)
  put(O/'STATUS.json',dict(status='RUNNING',current=name,recovery=True,pending=['uniform'] if name!='uniform' else []))
  with (O/(name+'_recovery.log')).open('x') as log:
   child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(seconds),*command],cwd=M,stdout=log,stderr=subprocess.STDOUT)
   put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),worker_pid=child.pid,current=name,seed=101));code=child.wait()
  if code:put(O/'STATUS.json',dict(status='FAILED_PRESERVED',current=name,returncode=code,recovery=True));return
 put(O/'STATUS.json',dict(status='COMPLETE',completed=['learned','uniform'],recovered_metadata_only=True,no_learned_retraining=True));s.report()
 put(R/'RECOVERY_COMPLETE.json',dict(status='COMPLETE',no_learned_retraining=True,uniform_run_once=True))
if __name__=='__main__':
 if '--evaluate' in sys.argv:evaluation()
 else:controller()
