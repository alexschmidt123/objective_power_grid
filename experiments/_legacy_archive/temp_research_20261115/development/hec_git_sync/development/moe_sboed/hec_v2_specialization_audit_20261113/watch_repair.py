"""MacBook completion watcher: finalize on LabPC and use approved safe sync workflow."""
import argparse,json,os,pathlib,shlex,subprocess,sys,time


def remote(command):
    p=subprocess.run(['ssh','-q','-o','BatchMode=yes','labpc',command],text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    return p.stdout


def read_status():
    code="import json,pathlib; p=pathlib.Path.home()/'Documents/objective_power_grid_temp/scratch_runs/hec_v2_E4_refresh_20261113'; f=p/'COMPLETION.json'; g=p/'STATUS.json'; print('CT_STATUS '+json.dumps(json.loads(f.read_text()) if f.exists() else json.loads(g.read_text()) if g.exists() else {'status':'STARTING'}))"
    output=remote('python3 -c '+shlex.quote(code))
    rows=[line[len('CT_STATUS '):] for line in output.splitlines() if line.startswith('CT_STATUS ')]
    if not rows:raise RuntimeError('Missing remote status receipt')
    return json.loads(rows[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--sync-script',type=pathlib.Path,default=pathlib.Path.home()/'scratch_runs/moe_workflow_sync_20261109/moe_sync.py')
    p.add_argument('--preflight',action='store_true');p.add_argument('--daemon',action='store_true')
    a=p.parse_args();out=a.output.expanduser().resolve()
    if a.preflight:
        print(json.dumps(dict(status='PREFLIGHT_OK',remote=read_status())));return
    if a.daemon:
        out.mkdir(parents=True,exist_ok=True)
        log=out/'watcher.log'
        if log.exists():raise FileExistsError('Preserve earlier watcher; no duplicate')
        arguments=[v for v in sys.argv[1:] if v!='--daemon']
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-u',str(pathlib.Path(__file__).resolve()),*arguments],
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'WATCHER_PID.json').write_text(json.dumps(dict(pid=child.pid,time=time.time()),indent=2))
        print(json.dumps(dict(status='WATCHER_STARTED',pid=child.pid,log=str(log))));return
    out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+16200;errors=0
    try:
        while time.monotonic()<deadline:
            try:status=read_status();errors=0
            except Exception as exc:
                errors+=1;print('POLL_ERROR',str(exc),flush=True)
                if errors>=3:raise
                time.sleep(30);continue
            (out/'LATEST_STATUS.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(status),flush=True)
            if status['status']=='COMPLETE':break
            if status['status'] in ['FAILED','PRESERVATION_FAILED','INCOMPLETE']:raise RuntimeError('Scientific run failed: '+json.dumps(status))
            time.sleep(30)
        else:raise TimeoutError('Completion watch timed out; running jobs untouched')
        command="PYTHONDONTWRITEBYTECODE=1 /home/grads/g/g.lin/miniconda3/envs/dellma/bin/python /home/grads/g/g.lin/Documents/objective_power_grid_temp/development/hec_git_sync/development/moe_sboed/hec_v2_specialization_audit_20261113/finalize_repair.py"
        print(remote(command),flush=True)
        files=['reports/hec_v2_E4_refresh_20261113/RESULTS.md','reports/hec_v2_E4_refresh_20261113/SUMMARY.json']
        receipt=out/'RESULTS_SYNC.json'
        process=subprocess.run([sys.executable,str(a.sync_script),'--message','Record matched V2 E4 adapter refresh results',
                                '--files',*files,'--receipt',str(receipt)],text=True,capture_output=True,timeout=180)
        print(process.stdout,flush=True)
        if process.returncode:raise RuntimeError('Safe synchronization failed; reports preserved: '+process.stderr)
        verified=json.loads(receipt.read_text());assert verified['status']=='VERIFIED'
        (out/'WATCHER_COMPLETION.json').write_text(json.dumps(dict(status='COMPLETE',sync=verified),indent=2))
    except Exception as exc:
        (out/'WATCHER_COMPLETION.json').write_text(json.dumps(dict(status='BLOCKED',error=str(exc),jobs_untouched=True),indent=2))
        raise


if __name__=='__main__':main()
