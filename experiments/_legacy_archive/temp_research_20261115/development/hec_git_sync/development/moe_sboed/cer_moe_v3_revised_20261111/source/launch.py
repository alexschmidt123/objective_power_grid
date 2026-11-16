"""Duplicate-safe launch of the frozen one-candidate/one-control pipeline."""
import os,pathlib,json,fcntl,subprocess,time,sys,hashlib
O=pathlib.Path(os.environ['CER_ROOT']);guard=(O/'.launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=O/'evidence/LAUNCH_RECEIPT.json';assert not receipt.exists()
manifest=json.loads((O/'PRETRAIN_MANIFEST.json').read_text());assert manifest['status']=='LOCKED_BEFORE_TRAINING'
for name,h in manifest['files'].items():assert hashlib.sha256((O/name).read_bytes()).hexdigest()==h,name
with (O/'TRAINING_LOGS/experiment.log').open('xb') as log:
    child=subprocess.Popen([sys.executable,'-u',str(O/'source/driver.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'CER_ROOT':str(O),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
receipt.write_text(json.dumps(dict(pid=child.pid,time=time.time(),git_commit=manifest['git_commit'],status_file=str(O/'evidence/EXPERIMENT_COMPLETION.json')),indent=2))
print(receipt.read_text())
