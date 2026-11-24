"""One durable required-job queue;no removed ablations and no duplicate retries."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
M=pathlib.Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/hec_v3_study_20261115';C=M/'configs/moe_sboed_resume_20261115.json';PY='/home/grads/g/g.lin/miniconda3/envs/dellma/bin/python';W=M/'tools/moe_sboed_benchmark_worker.py'
def put(path,v):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(path)
def read(path):return json.loads(path.read_text())
def execute(command,log,seconds):
 assert seconds>60
 with log.open('x') as f:
  p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(int(seconds-30)),PY,'-u',str(W),*command],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  put(O/'BENCHMARK_ACTIVE_PROCESS.json',dict(pid=p.pid,command=command,started=time.time(),hard_seconds=seconds))
  return p.wait()
def wait_gpu(deadline=None):
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  if deadline is not None and time.monotonic()>deadline:return False
  put(O/'BENCHMARK_LIVE_STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB'));time.sleep(20)
 return True
def main():
 locks=[]
 for name in ['.campaign.lock','.required_benchmark.lock']:
  f=(O/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 cfg=read(C);receipt=O/'REQUIRED_BENCHMARK_LAUNCH.json'
 attempt=dict(pid=os.getpid(),started=time.time(),protocol=str(C),queue=cfg['queue'])
 if not receipt.exists():put(receipt,attempt)
 else:
  with (O/'REQUIRED_CONTROLLER_RESUMES.jsonl').open('a') as f:f.write(json.dumps(attempt)+'\n')
 completed=[];failed=[]
 for job in cfg['queue']:
  arm,seed=job['arm'],job['seed'];root=O/f'seed{seed}_{arm}';result=root/'RESULTS.json';started=time.monotonic()
  if result.exists() and read(result)['status']=='COMPLETE':
   completed.append(dict(**job,status='REUSED_COMPLETE'));continue
  if root.exists():failed.append(dict(**job,status='PRESERVED_INCOMPLETE_NO_RETRY'));continue
  wait_gpu();put(O/'BENCHMARK_LIVE_STATUS.json',dict(status='TRAINING',current=job,completed=completed,failed=failed,pending=cfg['queue'][cfg['queue'].index(job)+1:]))
  started=time.monotonic()
  code=execute(['--task','train','--seed',str(seed),'--arm',arm],O/f'{job["method"]}_seed{seed}.log',14400)
  if code or not result.exists() or read(result)['status']!='COMPLETE':
   failed.append(dict(**job,status='FAILED_OR_CAPPED',returncode=code));continue
  if arm=='learned':
   remaining=14400-(time.monotonic()-started)
   if remaining>60:
    if not wait_gpu(started+14340):
     failed.append(dict(**job,status='EXPERT_EVALUATION_WAIT_CAP'));continue
    remaining=14400-(time.monotonic()-started)
    if remaining<=60:
     failed.append(dict(**job,status='EXPERT_EVALUATION_CAP_NO_EXTENSION'));continue
    put(O/'BENCHMARK_LIVE_STATUS.json',dict(status='EVALUATING_EXTRACTED_EXPERTS',current=job,remaining_seconds=remaining,completed=completed,failed=failed))
    code=execute(['--task','experts','--seed',str(seed),'--seconds',str(remaining)],O/f'MoE-sBOED-E1-E4_seed{seed}.log',remaining)
    if code:failed.append(dict(**job,status='EXPERT_EVALUATION_INCOMPLETE',returncode=code))
   else:failed.append(dict(**job,status='EXPERT_EVALUATION_CAP_NO_EXTENSION'))
  completed.append(dict(**job,status='COMPLETE'))
  put(O/'REQUIRED_BENCHMARK_PROGRESS.json',dict(completed=completed,failed=failed))
 put(O/'BENCHMARK_LIVE_STATUS.json',dict(status='INTERNAL_RUNS_COMPLETE' if not failed else 'INTERNAL_RUNS_FINISHED_WITH_FAILURES',completed=completed,failed=failed,next='External evaluation queue;RL native capped reproduction pending implementation validation'))
 # Already verified external checkpoints require evaluation,not duplicate training.
 external=M/'experiments/location_finding/moe_sboed_location_benchmark_20261115/EVALUATION_LAUNCH.json'
 if not external.exists():subprocess.run([PY,str(M/'tools/moe_sboed_external_evaluation.py'),'--launch'],check=True)
if __name__=='__main__':main()
