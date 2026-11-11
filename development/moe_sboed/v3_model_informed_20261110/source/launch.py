"""One duplicate-safe detached launch of the locked two-arm pilot."""
import os,sys,pathlib,json,subprocess,time,fcntl
O=pathlib.Path(__file__).resolve().parents[1];lock=(O/'.launch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);receipt=O/'evidence/LAUNCH_RECEIPT.json';assert not receipt.exists(),'No repeat launch';m=json.loads((O/'PRETRAIN_MANIFEST.json').read_text());assert m['status']=='LOCKED_BEFORE_PREPARATION_AND_TRAINING';env={**os.environ,'MOE_V3_RUN_ROOT':str(O),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'}
with (O/'TRAINING_LOGS/experiment.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-u',str(O/'source/driver.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
receipt.write_text(json.dumps(dict(pid=p.pid,process_group=p.pid,time=time.time(),git_commit=m['git_commit'],log=str(O/'TRAINING_LOGS/experiment.log'),two_fixed_arms=True),indent=2));print(receipt.read_text())
