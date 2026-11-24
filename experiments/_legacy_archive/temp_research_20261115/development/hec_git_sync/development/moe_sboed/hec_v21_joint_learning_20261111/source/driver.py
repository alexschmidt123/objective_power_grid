import threading,signal,subprocess,resource,gc
from common import *
from train import run_pair
from evaluation import run_evaluation

def telemetry(stop):
 previous=time.monotonic();cpu0=time.process_time()
 while not stop.wait(10):
  try:
   now=time.monotonic();cpu=time.process_time();v=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',');rss=int(pathlib.Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE');ru=resource.getrusage(resource.RUSAGE_CHILDREN);row=dict(time=time.time(),CPU_percent_one_core=100*(cpu-cpu0)/(now-previous),RSS_bytes=rss,children_cumulative_CPU_seconds=ru.ru_utime+ru.ru_stime,GPU_percent=float(v[0]),GPU_memory_MiB=float(v[1]),GPU_power_W=float(v[2]));previous=now;cpu0=cpu
   with (O/'TRAINING_LOGS/resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  except Exception as exc:
   with (O/'TRAINING_LOGS/telemetry_errors.log').open('a') as f:f.write(str(exc)+'\n')

def main():
 assert not (O/'evidence/EXPERIMENT_STARTED.json').exists();manifest=json.loads((O/'PRETRAIN_MANIFEST.json').read_text())
 for p,h in manifest['files'].items():assert sha(O/p)==h,p
 refs=json.loads((O/'evidence/PROTECTED_INPUTS_BEFORE.json').read_text())
 for p,h in refs.items():assert sha(p)==h,p
 original.setup()
 while True:
  try:leases=original.acquire();break
  except BlockingIOError:
   put(O/'QUEUE_STATUS.json',dict(status='WAITING_FOR_IDLE_GPU',time=time.time(),pid=os.getpid()));time.sleep(30)
 start=time.monotonic();stop=threading.Event();threading.Thread(target=telemetry,args=(stop,),daemon=True).start();put(O/'evidence/EXPERIMENT_STARTED.json',dict(time=time.time(),pid=os.getpid(),GPU=torch.cuda.get_device_name(),manifest_sha256=sha(O/'PRETRAIN_MANIFEST.json')))
 def watchdog():
  if not stop.wait(CFG['driver_TERM_seconds']):os.kill(os.getpid(),signal.SIGTERM)
  if not stop.wait(CFG['driver_hard_seconds']-CFG['driver_TERM_seconds']):os._exit(124)
 threading.Thread(target=watchdog,daemon=True).start();status='INCOMPLETE'
 try:
  if run_pair(start) and time.monotonic()-start<CFG['training_grace_seconds']:
   run_evaluation(start);status='COMPLETE' if json.loads((O/'evaluation/COMPLETION.json').read_text())['status']=='COMPLETE' else 'EVALUATION_INCOMPLETE'
 except Exception as exc:
  import traceback
  traceback.print_exc();put(O/'evidence/EXPERIMENT_ERROR.json',dict(type=type(exc).__name__,message=str(exc)));status='BLOCKED_CORRECTNESS_OR_RUNTIME'
 finally:
  stop.set();put(O/'evidence/EXPERIMENT_COMPLETION.json',dict(status=status,seconds=time.monotonic()-start,no_additional_training=True))
  try:
   import reports
   reports.write(O)
   for p,h in refs.items():assert sha(p)==h,p
   put(O/'evidence/PRESERVATION_VERIFIED.json',dict(status='PASS',protected_files=len(refs)));files={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['ARTIFACT_SHA256.json','experiment.log']};put(O/'ARTIFACT_SHA256.json',dict(status=status,files=files))
  except Exception as exc:
   import traceback
   traceback.print_exc();put(O/'evidence/REPORTING_ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
  print('EXPERIMENT_END',status,flush=True)
if __name__=='__main__':main()
