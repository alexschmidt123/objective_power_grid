"""Launch exactly one persistent queue without touching upstream jobs."""
import os,sys,pathlib,json,fcntl,subprocess,time
Q=pathlib.Path(__file__).resolve().parents[1];guard=(Q/'.launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);receipt=Q/'LAUNCH_RECEIPT.json';assert not receipt.exists(),'Queue already launched;inspect STATUS.json';m=json.loads((Q/'SOURCE_LOCK.json').read_text());assert m['git_commit'];assert json.loads((Q/'CPU_VALIDATION.json').read_text())['status']=='PASS'
with (Q/'queue.log').open('xb') as out:p=subprocess.Popen([sys.executable,'-u',str(Q/'source/queue_eval.py')],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
receipt.write_text(json.dumps(dict(pid=p.pid,process_group=p.pid,time=time.time(),git_commit=m['git_commit'],status_file=str(Q/'STATUS.json')),indent=2));print(receipt.read_text())
