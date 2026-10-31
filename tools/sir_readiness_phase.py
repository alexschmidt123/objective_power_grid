"""One readiness campaign, nonrepeatable phases, aggregate GPU lock cap1800s."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scratch_runs/10272026/10272026_sir_fourhour_readiness'
p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
lock=open(ROOT.with_name('objective_power_grid')/'.gpu0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
meter=BASE/'gpu_readiness_meter.json'
d=json.loads(meter.read_text()) if meter.exists() else dict(cap_gpu_locked_seconds=1800.,phases=[])
if any(x['phase']==a.phase for x in d['phases']):raise RuntimeError('Readiness phase already attempted; no automatic repetition')
used=sum(x['gpu_locked_seconds'] for x in d['phases']);remaining=1800-used
if remaining<=0:raise RuntimeError('Aggregate readiness cap reached')
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
cmd=[x.replace('{DEADLINE}',str(time.monotonic()+remaining-5)) for x in cmd]
row=dict(phase=a.phase,command=cmd,start_monotonic=time.monotonic(),pid=None,status='RUNNING',gpu_locked_seconds=0.)
d['phases'].append(row)
def save():
 tmp=meter.with_suffix('.partial');tmp.write_text(json.dumps(d,indent=2));tmp.replace(meter)
save()
with (BASE/(a.phase+'.log')).open('x') as log:
 proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled'))
 row['pid']=proc.pid;save()
 while proc.poll() is None:
  row['gpu_locked_seconds']=time.monotonic()-row['start_monotonic'];save()
  if row['gpu_locked_seconds']>=remaining-2:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=1)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
   row['deadline_terminated']=True;break
  time.sleep(.5)
 proc.wait()
row.update(gpu_locked_seconds=time.monotonic()-row['start_monotonic'],status='COMPLETE' if proc.returncode==0 else 'FAILED',returncode=proc.returncode)
d['total_gpu_locked_seconds']=sum(x['gpu_locked_seconds'] for x in d['phases']);save()
print(json.dumps(d),flush=True);raise SystemExit(proc.returncode)
