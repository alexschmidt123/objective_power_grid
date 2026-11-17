"""MacBook-only durable monitor;does not run,stop or restart GPU jobs."""
from pathlib import Path
import fcntl,json,os,shlex,subprocess,time,sys
D=Path(__file__).resolve().parent
BRANCH='research/hec-v3-routing-validation-20261115'
ROOT='/home/grads/g/g.lin/Documents/objective_power_grid'


def call(host,command):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    if r.returncode:raise RuntimeError(host+':'+r.stderr[-2000:]+r.stdout[-2000:])
    return r.stdout


def main():
    lock=(D/'.watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (D/'WATCH_PID.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time()))+'\n')
    while True:
        try:
            program='from pathlib import Path;import json;p=Path('+repr(ROOT+'/experiments/location_finding/moe_sboed_v4_replication_20261117/STATUS.json')+');print(p.read_text() if p.exists() else "{}")'
            status=json.loads(call('labpc','python3 -c '+shlex.quote(program)))
            (D/'LIVE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
            if status.get('status') in ['COMPLETE','FINISHED_WITH_FAILURES','TERMINATED_BY_USER']:
                output=call('labpc','env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/grads/g/g.lin/miniconda3/envs/dellma/bin/python '+ROOT+'/tools/moe_sboed_v4_replication_publish.py')
                (D/'PUBLISH_OUTPUT.txt').write_text(output)
                commit=call('labpc','git -C '+ROOT+' rev-parse HEAD').strip()
                worker='~/Documents/objective_power_grid_temp/development/hec_worker_sync'
                command='cd '+worker+' && test -z "$(git status --porcelain)" && git pull --ff-only --quiet origin '+BRANCH+' && git rev-parse HEAD && git ls-remote origin refs/heads/'+BRANCH
                values=call('ubuntu-pc',command).split()
                assert values[0]==values[1]==commit,'LabPC/GitHub/Ubuntu commit mismatch'
                local=D/'completed_report';local.mkdir(exist_ok=True)
                r=subprocess.run(['scp','-r','labpc:'+ROOT+'/reports/moe_sboed_v4_replication_20261117/.',str(local)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
                assert r.returncode==0,r.stderr
                (D/'SYNC_COMPLETE.json').write_text(json.dumps(dict(status=status['status'],commit=commit,worker_commit=values[0],github_commit=values[1],completed=time.time()),indent=2)+'\n')
                return
        except Exception as e:
            with (D/'WATCH_ERRORS.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),error=str(e)))+'\n')
        time.sleep(60)


if __name__=='__main__':
    if '--launch' in sys.argv:
        with (D/'watch.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(watcher_pid=proc.pid)))
    else:main()
