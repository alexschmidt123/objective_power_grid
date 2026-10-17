import os,json,time,subprocess,signal,hashlib,shutil
from pathlib import Path
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_moe_st_routing_dev_103_105';S=C/'source'
def read(p):return json.loads(p.read_text())
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 for rel,h in read(C/'source_manifest.json')['sha256'].items():assert sha(S/rel)==h,rel
 for rec in read(C/'original_checkpoint_manifest.json').values():assert sha(Path(rec['path']))==rec['sha256']
 if shutil.disk_usage(R).free<8*1024**3:raise RuntimeError('8GiB disk safety floor')
def main():
 assert read(C/'correctness.json')['status']=='passed' and read(C/'freeze.json')['status']=='frozen'
 for seed in [103,105]:
  guard();assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
  out=S/f'experiments/10172026/10172026_moe_st_dev/seed{seed}'
  cmd=['bash','run.sh','--config',str(S/'configs/moe_st_development.yaml'),'--journal','--variant','moe','--seed',str(seed),'--eval-seeds','1001','--eig-only','--source-check-manifest',str(C/'source_manifest.json'),'--output',str(out)]
  write(C/'commands'/f'seed{seed}.json',dict(command=cmd,cwd=str(S)))
  write(C/'status.json',dict(state='training_and_native_hard_evaluation',method='moe',training_routing_mode='straight_through',seed=seed,run=str(out)))
  started=time.monotonic()
  with (C/'logs'/f'seed{seed}.log').open('w') as f:
   proc=subprocess.Popen(cmd,cwd=S,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   write(C/'active_process.json',dict(pid=proc.pid,seed=seed,start=time.time()))
   try:rc=proc.wait(timeout=10800)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);raise RuntimeError('3h phase safety cap; partial output preserved')
  if rc:raise RuntimeError(f'Seed{seed} root experiment failed; retained; no automatic retry')
  tr=read(out/'models/moe_training.json')
  assert tr['generated_training_trajectories']==1000000 and tr['selected_checkpoint']['episodes']==1000000
  assert tr['settings']['training_routing_mode']=='straight_through' and tr['checkpoint_selection']=='final_predeclared_fixed_budget'
  write(C/'completion'/f'seed{seed}.json',dict(status='complete',seed=seed,run=str(out),primary_training_episodes=1000000,wall_seconds=time.monotonic()-started,checkpoint_sha256=sha(out/'models/moe.pt'),summary=read(out/'summary.json'),primary_result_replaced=False,reserved_final_seed_used=False))
  guard()
 write(C/'status.json',dict(state='training_complete_starting_paired_analysis',seeds=[103,105]))
 script=C/'scripts/evaluate_campaign.py'
 assert script.exists(),'Paired analysis script must be prepared before training completes'
 with (C/'logs/paired_analysis.log').open('w') as f:
  rc=subprocess.call(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(script)],cwd=S,stdout=f,stderr=subprocess.STDOUT)
 if rc:raise RuntimeError('Paired analysis failed; training outputs preserved')
 guard();write(C/'status.json',dict(state='complete',seeds=[103,105],report=str(C/'reports/report.md'),primary_results_replaced=False,reserved_final_seed_used=False))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback;write(C/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));write(C/'status.json',dict(state='failed_preserved',error=str(exc)));raise
