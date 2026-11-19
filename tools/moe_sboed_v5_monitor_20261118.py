"""Wait for the one existing pilot; publish only its verified terminal reports."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v5_20261118'
lock=(O/'.publisher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def receipt(value):(O/'PUBLICATION_STATUS.json').write_text(json.dumps(value,indent=2)+'\n')
receipt(dict(status='WAITING',pid=os.getpid()))
for _ in range(960):
    state=json.loads((O/'STATUS.json').read_text())
    if state['status'] in ['COMPLETE','FAILED_OR_CAPPED']:
        receipt(dict(status='VERIFYING_AND_PUBLISHING',pid=os.getpid(),controller=state))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        result=subprocess.run([sys.executable,'-B','-u',str(M/'tools/moe_sboed_v5_publish_20261118.py')],cwd=M,env=env)
        receipt(dict(status='PUBLISHED' if result.returncode==0 else 'PUBLICATION_FAILED_OR_CAPPED',pid=os.getpid(),returncode=result.returncode))
        break
    time.sleep(45)
else:receipt(dict(status='MONITOR_TIME_LIMIT_NO_JOB_CHANGE',pid=os.getpid()))
