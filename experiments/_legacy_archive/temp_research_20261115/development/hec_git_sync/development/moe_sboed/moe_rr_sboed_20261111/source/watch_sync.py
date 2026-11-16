"""MacBook completion watcher: report-only LabPC writes, then approved sync.

No training launch/restart, SSH keys, Ubuntu computation or source mutation.
"""
import pathlib,subprocess,json,time,sys,fcntl,os
TASK='development/moe_sboed/moe_rr_sboed_20261111'
LOCAL=pathlib.Path.home()/'scratch_runs/moe_rr_sboed_20261111'
SYNC=pathlib.Path.home()/'scratch_runs/moe_workflow_sync_20261109/moe_sync.py'
POLL=r'''
import pathlib,json
b=pathlib.Path.home()/'Documents/objective_power_grid_temp';o=b/'scratch_runs/moe_rr_sboed_20261111'
def read(name):
 p=o/name
 return json.loads(p.read_text()) if p.exists() else None
done=read('evidence/EXPERIMENT_COMPLETION.json');receipt=read('evidence/LAUNCH_RECEIPT.json')
active=False
if receipt:
 p=pathlib.Path('/proc')/str(receipt['pid'])/'cmdline'
 active=p.exists() and str(o/'source/driver.py').encode() in p.read_bytes()
print(json.dumps(dict(done=done,active=active,launch=receipt,queue=read('QUEUE_STATUS.json'),
 preparation=read('TRAINING_LOGS/preparation_status.json'),full=read('TRAINING_LOGS/full_status.json'),control=read('TRAINING_LOGS/control_status.json'))))
'''
PUBLISH=r'''
import pathlib,json,subprocess,importlib.util,hashlib,sys
sys.dont_write_bytecode=True
b=pathlib.Path.home()/'Documents/objective_power_grid_temp';o=b/'scratch_runs/moe_rr_sboed_20261111'
repo=b/'development/hec_git_sync';target=repo/'development/moe_sboed/moe_rr_sboed_20261111'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Dirty authoritative worktree preserved;report sync blocked'
assert subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()=='research/hec-moe-sboed-20261109'
manifest=json.loads((o/'PRETRAIN_MANIFEST.json').read_text())
for name,h in manifest['files'].items():assert hashlib.sha256((o/name).read_bytes()).hexdigest()==h,name
assert json.loads((o/'evidence/PRESERVATION_VERIFIED.json').read_text())['status']=='PASS'
spec=importlib.util.spec_from_file_location('frozen_cer_reports',o/'source/reports.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.write(o,target)
done=json.loads((o/'evidence/EXPERIMENT_COMPLETION.json').read_text())
(target/'RUN_STATUS.md').write_text('# RR-MoE run status\n\n'+done['status']+'\n\nSource lock commit: '+manifest['git_commit']+'\n\nSee PAIRED_EIG_COMPARISON.md and RESULTS.json for completed,negative or incomplete evidence.\n')
print(json.dumps(dict(status=done['status'],files=[*m.DYNAMIC,'RESULTS.json','RUN_STATUS.md'])))
'''

def remote(source):
    process=subprocess.run(['ssh','-q','-o','BatchMode=yes','labpc','python3 -'],input=source,text=True,capture_output=True,timeout=90)
    if process.returncode:raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    return json.loads(process.stdout)

def main():
    LOCAL.mkdir(parents=True,exist_ok=True);guard=(LOCAL/'.watch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (LOCAL/'FINAL_REPORT_SYNC.json').exists(),'Completion already synchronized'
    errors=0;start=time.monotonic()
    while time.monotonic()-start<52000:
        try:
            status=remote(POLL)
            (LOCAL/'WATCH_STATUS.json').write_text(json.dumps(dict(time=time.time(),**status),indent=2)+'\n')
            if status['done']:
                # Driver writes the terminal receipt before preservation/reporting;
                # wait for owned process exit before publishing immutable outputs.
                if status['active']:time.sleep(15);continue
                published=remote(PUBLISH)
                files=[TASK+'/'+name for name in published['files']]
                subprocess.run([sys.executable,str(SYNC),'--message','Report RR-MoE matched Top2 EIG and specialist correction evidence','--files',*files,'--receipt',str(LOCAL/'FINAL_REPORT_SYNC.json')],check=True)
                for name in published['files']:
                    subprocess.run(['scp','-q','labpc:Documents/objective_power_grid_temp/development/hec_git_sync/'+TASK+'/'+name,str(LOCAL/name)],check=True)
                print('RR_REPORTS_SYNCHRONIZED',published['status'],flush=True);return
            if status['launch'] and not status['active']:raise RuntimeError('Driver exited without a terminal receipt; no automatic restart')
            errors=0
        except Exception as exc:
            errors+=1;print('WATCH_ERROR',errors,str(exc),flush=True)
            (LOCAL/'WATCH_ERROR.json').write_text(json.dumps(dict(time=time.time(),errors=errors,message=str(exc)),indent=2))
            if errors>=3:raise
        time.sleep(30)
    raise RuntimeError('Completion watcher time limit reached; no training restart or extension')

if __name__=='__main__':main()
