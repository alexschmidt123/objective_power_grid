"""Explicit authorized launch;existing output roots never imply a restart."""
import json,os,pathlib,subprocess,sys,time
M=pathlib.Path.home()/'Documents/objective_power_grid';D=pathlib.Path(__file__).resolve().parent;O=M/'experiments/location_finding/hec_v3_study_20261115'
def main():
 receipt=O/'LAUNCH.json';assert not receipt.exists(),'Duplicate/restart forbidden'
 branch=subprocess.check_output(['git','-C',str(M),'branch','--show-current'],text=True).strip();assert branch=='research/hec-v3-routing-validation-20261115'
 assert not subprocess.check_output(['git','-C',str(M),'status','--porcelain','--','src/research/hec_v3_study'],text=True).strip(),'Commit/synchronize source before launch'
 with (O/'controller.log').open('x') as log:
  p=subprocess.Popen([sys.executable,'-u',str(D/'controller.py')],cwd=str(D),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 value=dict(status='LAUNCHED',pid=p.pid,started=time.time(),source=str(M),branch=branch,commit=subprocess.check_output(['git','-C',str(M),'rev-parse','HEAD'],text=True).strip(),sequential_training_runs=25,per_run_hard_seconds=14400,no_automatic_retries=True)
 receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
