"""Single-owner sequential queue;four-hour watchdog per GPU run;no automatic retries."""
import fcntl,hashlib,json,os,pathlib,subprocess,sys,time
MAIN=pathlib.Path.home()/'Documents/objective_power_grid';HERE=pathlib.Path(__file__).resolve().parent;OUT=MAIN/'experiments/location_finding/hec_v3_study_20261115';REPORT=MAIN/'reports/hec_v3_study_20261115'
P=json.loads((HERE/'protocol.json').read_text());PY='/home/grads/g/g.lin/miniconda3/envs/dellma/bin/python'
def put(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def sources():return {str(p.relative_to(MAIN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()}
def wait_idle():
 while True:
  if (OUT/'STOP_AFTER_CURRENT').exists():raise RuntimeError('User stop marker;no next job launched')
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if not active:return
  put(OUT/'CAMPAIGN_STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB',active_pids=active));time.sleep(30)
def run(command,label,destination):
 wait_idle();assert not destination.exists(),destination
 put(OUT/'CAMPAIGN_STATUS.json',dict(status='RUNNING',current=label,destination=str(destination),required_training_runs=25))
 with (OUT/(label+'.log')).open('x') as log:
  p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=60','14340',PY,'-u',str(HERE/'child.py'),*command],cwd=str(HERE),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  put(OUT/'ACTIVE_PROCESS.json',dict(pid=p.pid,label=label,hard_seconds=14400,started=time.time()))
  code=p.wait()
 result=destination/('STATUS.json' if label=='frozen_diagnosis' else 'RESULTS.json')
 if code or not result.exists() or json.loads(result.read_text())['status']!='COMPLETE':raise RuntimeError(f'{label}:exit={code};partial evidence retained;no retry')
 subprocess.run([PY,str(HERE/'analysis.py'),'--publish'],check=True,cwd=str(HERE))
def main():
 lock=(OUT/'.campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((REPORT/'NATIVE_VALIDATION.json').read_text())['status']=='PASS'
 assert json.loads((REPORT/'COMPLETED_V3_AUDIT.json').read_text())['status']=='VERIFIED'
 before=sources();put(OUT/'FROZEN_SOURCE_MANIFEST.json',before)
 try:
  run(['--task','diagnose'],'frozen_diagnosis',OUT/'completed_v3_diagnosis');assert sources()==before
  for entry in P['queue']:
   seed,arm=entry['seed'],entry['arm'];run(['--task','train','--seed',str(seed),'--arm',arm],f'seed{seed}_{arm}',OUT/f'seed{seed}_{arm}');assert sources()==before
  put(OUT/'CAMPAIGN_STATUS.json',dict(status='COMPLETE',completed_training_runs=25,protocol='protocol.json',scientific_claims='See seed-level statistics;completion is not evidence of superiority'))
 except Exception as e:put(OUT/'CAMPAIGN_STATUS.json',dict(status='BLOCKED',error=repr(e),automatic_retry=False));raise
 finally:lock.close()
if __name__=='__main__':main()
