"""One owned fresh-training pair,one frozen evaluation,then reports and seal."""
import os,sys,time,threading,signal,traceback
sys.dont_write_bytecode=True
from native_common import *
from train_v2 import run_pair
from evaluate_v2 import run_evaluation
from seal import seal

def main():
 assert not (O/'evidence/EXPERIMENT_STARTED.json').exists(),'One experiment only;no resume';manifest=json.loads((O/'PRETRAIN_MANIFEST.json').read_text())
 for p,h in manifest['files'].items():assert sha(O/p)==h,p
 refs=json.loads((O/'evidence/REFERENCE_SOURCE_INPUTS.json').read_text())
 for p,h in refs.items():assert sha(p)==h,p
 original.setup()
 while True:
  try:leases=original.acquire();break
  except BlockingIOError:
   put(O/'QUEUE_STATUS.json',dict(status='WAITING_FOR_IDLE_GPU',time=time.time(),pid=os.getpid()));time.sleep(30)
 start=time.monotonic();put(O/'evidence/EXPERIMENT_STARTED.json',dict(pid=os.getpid(),time=time.time(),GPU=torch.cuda.get_device_name(),parent_native_iteration=14001,manifest_sha256=sha(O/'PRETRAIN_MANIFEST.json'),hard_seconds=33600));done=threading.Event()
 def watchdog():
  if not done.wait(max(0,33000-(time.monotonic()-start))):
   put(O/'evidence/WATCHDOG_TERM.json',dict(reason='original owned-experiment runtime cap',seconds=time.monotonic()-start));os.kill(os.getpid(),signal.SIGTERM)
   if not done.wait(max(0,33600-(time.monotonic()-start))):os._exit(124)
 threading.Thread(target=watchdog,daemon=True).start();status='INCOMPLETE'
 try:
  complete=run_pair(start)
  if complete and time.monotonic()-start<32400:
   run_evaluation(start);status='COMPLETE' if json.loads((O/'evaluation/COMPLETION.json').read_text())['status']=='COMPLETE' else 'EVALUATION_INCOMPLETE'
 except Exception as exc:
  traceback.print_exc();put(O/'evidence/EXPERIMENT_ERROR.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.monotonic()-start));status='BLOCKED'
 finally:
  put(O/'evidence/EXPERIMENT_COMPLETION.json',dict(status=status,seconds=time.monotonic()-start,additional_training_launched=False,Ubuntu_accessed=False,archive_transfer_controlled=False))
  try:
   import write_reports
   write_reports.write()
   seal()
  except Exception as exc:
   traceback.print_exc();put(O/'evidence/REPORTING_ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
  done.set();print('EXPERIMENT_END',status,flush=True)
if __name__=='__main__':main()
