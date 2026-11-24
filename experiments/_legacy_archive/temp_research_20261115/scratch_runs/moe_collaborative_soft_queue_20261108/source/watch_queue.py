"""Detached LabPC watchdog;strict dependency gate;single persistent launchclaim."""
import os,sys,json,time,pathlib,subprocess,hashlib,fcntl,traceback
O=pathlib.Path(__file__).resolve().parents[1];UP=O.parent/'moe_task_specialized_option_20261107';lock=(O/'QUEUE_PROCESS.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=json.loads((O/'CONFIG_LOCKED.json').read_text());start=time.monotonic()
def put(name,value):
 p=O/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def status(state,**extra):put('QUEUE_STATUS.json',dict(state=state,pid=os.getpid(),time=time.time(),wait_seconds=time.monotonic()-start,upstream=str(UP),**extra));print(state,json.dumps(extra),flush=True)
def verify_queue():
 manifest=json.loads((O/'QUEUE_MANIFEST.json').read_text())
 for name,h in manifest['files'].items():assert sha(O/name)==h,name
def read(p):
 try:return json.loads(p.read_text()) if p.exists() else None
 except (OSError,json.JSONDecodeError):return None
def upstream_gate():
 completion=read(UP/'evidence/EXPERIMENT_COMPLETION.json');error=read(UP/'evidence/EXPERIMENT_ERROR.json');reporterror=read(UP/'evidence/REPORTING_ERROR.json')
 if error or reporterror:return 'FAILED',dict(error=error,reporterror=reporterror)
 for arm in ['control','candidate']:
  result=read(UP/f'arms/{arm}/RESULT.json')
  if result and (result.get('status')!='COMPLETE' or result.get('native_iterations')!=10001):return 'FAILED',dict(arm=arm,result=result)
 if completion and completion.get('status')!='COMPLETE':return 'FAILED',completion
 if not completion:return 'WAIT',dict(reason='training/evaluation notcomplete')
 for name in ['arms/control/RESULT.json','arms/candidate/RESULT.json','evaluation/COMPLETION.json','evaluation/NATIVE_COMPETENCE_PANEL.json','evaluation/SPECIALIZATION_RESULTS.json','evaluation/EIG_RESULTS.json','evidence/FINAL_PRESERVATION.json','ARTIFACT_SHA256.json']:
  if read(UP/name) is None:return 'WAIT',dict(reason='missing completion evidence',file=name)
 for arm in ['control','candidate']:
  result=read(UP/f'arms/{arm}/RESULT.json');assert result['status']=='COMPLETE' and result['native_iterations']==10001
 if read(UP/'evaluation/COMPLETION.json')['status']!='COMPLETE' or read(UP/'evaluation/NATIVE_COMPETENCE_PANEL.json')['status']!='COMPLETE':return 'FAILED',dict(reason='upstreamevaluation incomplete')
 if read(UP/'evidence/FINAL_PRESERVATION.json')['status']!='VERIFIED':return 'FAILED',dict(reason='upstreampreservation failed')
 seal=read(UP/'ARTIFACT_SHA256.json');assert seal['status']=='VERIFIED'
 # Process-state check ignores favorable/unfavorable scientific outcomes.
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   except (OSError,PermissionError):continue
   if str(UP/'source/run_experiment.py') in cmd:return 'WAIT',dict(reason='upstream driver hasnotexited',pid=int(p.name))
 return 'READY',dict(completion=completion,scientific_outcome_gate=False)
def gpu_idle():
 result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert result.returncode==0,result.stderr;memory,util=[int(v.strip()) for v in result.stdout.strip().splitlines()[0].split(',')];apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert apps.returncode==0;apps=apps.stdout.strip();return memory<512 and util<=5 and not apps,dict(memory_MiB=memory,utilization_percent=util,compute_pids=apps)
def run_script(name):
 path=O/'source'/name;verify_queue();log=O/'queue_logs'/f'{name}.log'
 with log.open('ab') as f:
  p=subprocess.Popen([sys.executable,'-u',str(path)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
  put('queue_logs/child_status.json',dict(script=name,pid=p.pid,time=time.time()));rc=p.wait()
 assert rc==0,f'{name} exited{rc};see{log}'
def main():
 verify_queue();assert not (O/'QUEUE_LAUNCH_CLAIM.json').exists(),'Prior claim exists;noautomaticrestart';status('WAITING_UPSTREAM')
 while True:
  state,detail=upstream_gate()
  if state=='FAILED':status('BLOCKED_UPSTREAM_FAILURE',detail=detail);return
  if time.monotonic()-start>cfg['queue']['maximum_wait_seconds']:status('BLOCKED_QUEUE_TIMEOUT');return
  if state=='READY':
   idle,gpu=gpu_idle()
   if idle:
    # Require two clean samples;release noresources andinterrupt nojobs.
    time.sleep(5);idle,gpu=gpu_idle()
    if idle:break
   status('WAITING_GPU',gpu=gpu)
  else:status('WAITING_UPSTREAM',detail=detail)
  time.sleep(cfg['queue']['poll_seconds'])
 verify_queue()
 for arm in ['control','candidate']:
  result=read(UP/f'arms/{arm}/RESULT.json');assert result['native_optimizer_steps']==dict(actor=10001,critic=50005,posterior=1500150);assert sha(UP/f'arms/{arm}/actor.pt')==result['actor_sha256']
 eigen=read(UP/'evaluation/EIG_RESULTS.json');assert all(eigen['methods'][m]['n']==2000 for m in ['candidate','control','reference','vsoed'])
 fd=os.open(O/'QUEUE_LAUNCH_CLAIM.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),time=time.time(),upstream_completion=detail,single_attempt=True),f,indent=2)
 status('GPU_PREFLIGHT');run_script('gpu_preflight.py');status('LOCKING_PROTOCOL');run_script('lock_protocol.py')
 while True:
  idle,gpu=gpu_idle()
  if idle:break
  if time.monotonic()-start>cfg['queue']['maximum_wait_seconds']:raise TimeoutError('GPU unavailable after protocol lock')
  status('WAITING_GPU_BEFORE_TRAINING',gpu=gpu);time.sleep(cfg['queue']['poll_seconds'])
 status('RUNNING_EXPERIMENT');run_script('run_experiment.py');completion=read(O/'evidence/EXPERIMENT_COMPLETION.json');preservation=read(O/'evidence/FINAL_PRESERVATION.json')
 artifact=read(O/'ARTIFACT_SHA256.json');reporterror=read(O/'evidence/REPORTING_ERROR.json')
 if not completion or completion.get('status')!='COMPLETE' or not preservation or preservation.get('status')!='VERIFIED' or reporterror or not artifact or artifact.get('status')!='VERIFIED':status('BLOCKED_EXPERIMENT_FAILURE',completion=completion,preservation=preservation);return
 status('COMPLETE',completion=completion,preservation=preservation,additional_candidates=False)
def final_artifact_manifest():
 import shutil
 p=O/'ARTIFACT_SHA256.json'
 if p.exists():shutil.copyfile(p,O/'evidence/DRIVER_ARTIFACT_SHA256.json')
 files={str(v.relative_to(O)):sha(v) for v in O.rglob('*') if v.is_file() and v.name!='ARTIFACT_SHA256.json' and '__pycache__' not in v.parts}
 preservation=read(O/'evidence/FINAL_PRESERVATION.json')
 put('ARTIFACT_SHA256.json',dict(status='VERIFIED' if preservation and preservation.get('status')=='VERIFIED' else 'INCOMPLETE',phase='AFTER_ALL_CHILDREN_EXITED',files=files))

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  traceback.print_exc();status('BLOCKED_EXCEPTION',type=type(exc).__name__,message=str(exc))
 finally:
  try:final_artifact_manifest()
  except Exception:traceback.print_exc()
