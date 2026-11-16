"""Detach one authorized V3-only run behind a hard wall-clock watchdog."""
import json,os,pathlib,subprocess,time
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync';HERE=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'scratch_runs/hec_v3_fulltrain_20261114_run01'
def main():
    assert not OUT.exists(),'Never overwrite a run'
    assert not subprocess.check_output(['git','-C',str(WORK),'status','--porcelain'],text=True).strip(),'Commit and synchronize source first'
    git=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip()
    for name in ['validation01','smoke01']:
        result=json.loads((ROOT/('scratch_runs/hec_v3_fulltrain_20261114_'+name)/'RESULTS.json').read_text());assert result['status']=='COMPLETE'
    log=ROOT/'scratch_runs/hec_v3_fulltrain_20261114_run01.launch.log';receipt=ROOT/'scratch_runs/hec_v3_fulltrain_20261114_run01.LAUNCH.json'
    assert not log.exists() and not receipt.exists()
    command=['timeout','--signal=TERM','--kill-after=60','43140','/home/grads/g/g.lin/miniconda3/envs/dellma/bin/python','-u',str(HERE/'run_full.py'),'--output',str(OUT)]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUNBUFFERED']='1'
    with log.open('x') as f:p=subprocess.Popen(command,cwd=str(HERE),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    value=dict(status='LAUNCHED',pid=p.pid,wall_clock_start=time.time(),hard_max_seconds_including_kill_grace=43200,command=command,commit=git,output=str(OUT),log=str(log),baseline_training=False)
    receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
