"""One detached native V5-A worker; no queued second seed."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,os,signal,subprocess,time

def write(p,x):
 tmp=p.with_name(p.name+'.supervisor.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args();c=Path(a.campaign);root=c.parents[1]
 lock=(c/'supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=c/'seed101';command=['bash',str(root/'run.sh'),'--moe-sboed-v5a-training','--config',str(root/'configs/moe_sboed_v5a_train_20261031.yaml'),'--action','train','--seed','101','--output',str(out)]
 if out.exists():raise RuntimeError('Existing primary run; no duplicate or reset')
 entry=time.monotonic();env=dict(os.environ,BOED_V5A_ENTRY_MONOTONIC=str(entry),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
 with (c/'train_seed101.log').open('ab') as log:
  process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(c/'launched.json',dict(supervisor_pid=os.getpid(),worker_pid=process.pid,command=command,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),seed102='NOT_STARTED'))
  signalled=False
  while process.poll() is None:
   clock_path=out/'compute_clock.json';charged=time.monotonic()-entry;stopped=False
   if clock_path.exists():
    try:
     clock=json.loads(clock_path.read_text());stopped='stopped_at' in clock
     charged=clock['credit_seconds']+clock.get('stopped_at',time.monotonic())-clock['start']-sum(b-a for a,b,_ in clock['excluded'])
    except (OSError,ValueError,KeyError):pass
   if not stopped and charged>=14397 and not signalled:
    os.killpg(process.pid,signal.SIGTERM);signalled=True;write(c/'time_guard.json',dict(pid=process.pid,charged_seconds=charged,signal='SIGTERM'))
   if not stopped and charged>=14399 and process.poll() is None:
    os.killpg(process.pid,signal.SIGKILL);write(c/'time_guard.json',dict(pid=process.pid,charged_seconds=charged,signal='SIGKILL',status='PRESERVE_LAST_ELIGIBLE_CHECKPOINT'))
   status_path=out/'run_status.json'
   if status_path.exists():
    try:
     status=json.loads(status_path.read_text());m=status.get('last_metrics',{})
     temp=c/'compute_accounting.live.tmp'
     with temp.open('w',newline='') as f:
      w=csv.writer(f);w.writerow(['phase','seed','optimizer_updates','primary_training_trajectories','actual_simulator_particle_stages','conditional_seconds','charged_seconds','state']);w.writerow(['P1',101,status['update'],status['primary_training_trajectories'],m.get('actual_particle_stages',''),m.get('total_conditional_seconds',0),charged,status['state']])
     temp.replace(c/'compute_accounting.csv')
    except (OSError,ValueError,KeyError):pass
   time.sleep(.25)
  code=process.returncode
 write(c/'process_exit.json',dict(worker_pid=process.pid,returncode=code,seed102='NOT_STARTED',timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 if code==0 and (out/'paired_eig_seed101.json').exists():
  status=json.loads((out/'run_status.json').read_text());status['state']='COMPLETE_WITH_DEVELOPMENT_EVALUATION';status['evaluation_result']=str(out/'paired_eig_seed101.json');write(c/'run_status.json',status)
 elif code!=0:
  last=json.loads((c/'run_status.json').read_text()) if (c/'run_status.json').exists() else {}
  last.update(state='FAILED_OR_HARD_STOP',returncode=code,worker_pid=process.pid,seed102='NOT_STARTED');write(c/'run_status.json',last)

if __name__=='__main__':main()
