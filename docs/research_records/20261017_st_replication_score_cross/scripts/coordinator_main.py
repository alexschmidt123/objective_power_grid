import os,json,time,subprocess,signal,hashlib,shutil,traceback
from pathlib import Path
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_st_replication_score_reverse_cross';P=R/'experiments/10172026/10172026_moe_st_routing_dev_103_105'
PY='/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 for label in ['moe','score']:
  for rel,h in read(C/f'{label}_source_manifest.json')['sha256'].items():assert sha(C/f'{label}_source'/rel)==h,rel
 for p,h in read(C/'preserved_checkpoint_hashes.json').items():assert sha(Path(p))==h,p
 assert shutil.disk_usage(R).free>8*1024**3,'Disk safety floor'
def phase(name,cmd,cwd,timeout,kind):
 guard();assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU already busy; preserve queue without duplicate launch'
 marker=C/'completion'/f'{name}.json';assert not marker.exists(),'No automatic repeat'
 write(C/'commands'/f'{name}.json',dict(command=cmd,cwd=str(cwd)))
 write(C/'status.json',dict(state=kind,phase=name,started_utc=time.time(),completed=[p.stem for p in (C/'completion').glob('*.json')]))
 started=time.monotonic()
 with (C/'logs'/f'{name}.log').open('w') as f:
  p=subprocess.Popen(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(C/'active_process.json',dict(phase=name,pid=p.pid,started=time.time()))
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);raise RuntimeError(name+' safety timeout; partial outputs retained')
 assert rc==0,(name,rc)
 write(marker,dict(status='complete',wall_seconds=time.monotonic()-started,command=cmd));guard()
def main():
 assert read(C/'score_readiness.json')['status']=='passed'
 # Reuse original ST source without any scientific changes.
 for rel,h in read(P/'source_manifest.json')['sha256'].items():assert sha(C/'moe_source'/rel)==h,rel
 S=C/'score_source';out=(C/'runs/score102_policy103').resolve()
 phase('score102_policy103',['bash','run.sh','--config',str(S/'configs/score102_policy103.yaml'),'--journal','--methods','scorebed','--seed','103','--eval-seeds','1001','--eig-only','--source-check-manifest',str(C/'score_source_manifest.json'),'--output',str(out)],S,1800,'score_cross_training_and_evaluation')
 tr=read(out/'models/scorebed_training.json');assert tr['complete_model_training_episodes']==1000000 and tr['new_generated_training_trajectories']==297006 and tr['selected_checkpoint']['episodes']==1000000
 phase('score_cross_analysis',[PY,str(C/'scripts/score_analysis.py')],C,900,'score_cross_analysis')
 S=C/'moe_source'
 for seed in [101,102,104]:
  out=(C/f'runs/moe_seed{seed}').resolve()
  phase(f'moe_seed{seed}',['bash','run.sh','--config',str(S/'configs/moe_st_development.yaml'),'--journal','--variant','moe','--seed',str(seed),'--eval-seeds','1001','--eig-only','--source-check-manifest',str(C/'moe_source_manifest.json'),'--output',str(out)],S,10800,'moe_training_and_hard_evaluation')
  tr=read(out/'models/moe_training.json');assert tr['generated_training_trajectories']==1000000 and tr['selected_checkpoint']['episodes']==1000000 and tr['settings']['training_routing_mode']=='straight_through'
 phase('moe_five_seed_analysis',[PY,str(C/'scripts/moe_analysis.py')],S,1800,'moe_five_seed_analysis')
 phase('score_history_errors',[str(C/'score_source/.venv_scorebed/bin/python'),str(C/'scripts/score_errors.py')],C,1800,'score_history_errors')
 guard();write(C/'status.json',dict(state='complete',new_moe_seeds=[101,102,104],preserved_moe_seeds=[103,105],score_cross='score102_policy103',primary_results_replaced=False,reserved_final_seed_used=False))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(C/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));write(C/'status.json',dict(state='failed_preserved',error=str(exc)));raise
