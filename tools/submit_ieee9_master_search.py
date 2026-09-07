#!/usr/bin/env python3
"""Submit an immutable master-bank search on Grace; standard library only."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess,tarfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',type=Path,required=True);p.add_argument('--commit',required=True)
a=p.parse_args();runtime=Path('/scratch/user/g.lin/objective_power_grid_runtime')
stamp=datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
out=runtime/'experiments'/(stamp+'_ieee9_master_search_Uctrl_T5_Nobs5_sigma0p005')
source=out/'source_snapshot';source.mkdir(parents=True,exist_ok=False)
with tarfile.open(str(a.archive)) as t:t.extractall(str(source))
(source/'data').symlink_to(runtime/'data',target_is_directory=True)
previous=runtime/'experiments/09062026_222620_ieee9_stronger_duration_audit_Uctrl_T5_Nobs5_sigma0p005/campaign.json'
shutil.copy2(str(previous),str(out/'previous_campaign.json'))
files={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for base in ('src','tools','configs','hprc') for f in (source/base).rglob('*') if f.is_file()}
(out/'source_manifest.json').write_text(json.dumps(dict(commit=a.commit,files=files,archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest()),indent=2)+'\n')
jobs={};wrapper=source/'hprc/ieee9_master_search_stage.slurm'
def submit(stage,options):
 job=subprocess.check_output(['sbatch','--parsable','--job-name=ieee9-master-'+stage]+options+[str(wrapper),stage,str(out)],universal_newlines=True).strip().split(';')[0]
 if not job.isdigit():raise RuntimeError(job)
 jobs[stage]=job;(out/'slurm_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');print(stage,job,flush=True);return job
prep=submit('prepare',['--partition=gpu','--gres=gpu:a100:1','--time=12:00:00'])
fresh=submit('fresh',['--partition=gpu','--gres=gpu:a100:1','--time=02:00:00','--dependency=afterok:'+prep])
conf=submit('confirm',['--partition=gpu','--gres=gpu:a100:1','--time=12:00:00','--array=0-4%2','--dependency=afterok:'+fresh])
submit('finalize',['--partition=short','--time=00:15:00','--dependency=afterok:'+conf])
print('CAMPAIGN='+str(out),flush=True)
