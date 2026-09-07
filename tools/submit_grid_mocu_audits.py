#!/usr/bin/env python3
"""Submit frozen IEEE14/30 audit campaigns on Grace (standard library only)."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,tarfile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args();runtime=Path('/scratch/user/g.lin/objective_power_grid_runtime')
    stamp=datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
    all_jobs={}
    for system in ('ieee14','ieee30'):
        out=runtime/'experiments'/(stamp+'_'+system+'_duration_audit_Uctrl_T5_Nobs5_sigma0p005')
        source=out/'source_snapshot';source.mkdir(parents=True,exist_ok=False)
        with tarfile.open(str(args.archive)) as archive:archive.extractall(str(source))
        files={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file()}
        (out/'source_manifest.json').write_text(json.dumps(dict(commit=args.commit,files=files,archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest()),indent=2)+'\n')
        wrapper=source/'hprc/grid_mocu_audit_stage.slurm';config='configs/'+system+'_mocu.yaml'
        jobs={}
        def submit(stage,options):
            cmd=['sbatch','--parsable','--job-name='+system+'-audit-'+stage]+options+[str(wrapper),stage,str(out),config]
            job=subprocess.check_output(cmd,universal_newlines=True).strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Unexpected job ID '+job)
            jobs[stage]=job
            (out/'slurm_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
            print(system,stage,job,flush=True)
            return job
        prep=submit('prepare',['--partition=gpu','--gres=gpu:a100:1','--time=12:00:00'])
        fresh=submit('fresh',['--partition=gpu','--gres=gpu:a100:1','--time=02:00:00','--dependency=afterok:'+prep])
        conf=submit('confirm',['--partition=gpu','--gres=gpu:a100:1','--time=24:00:00','--array=0-5%2','--dependency=afterok:'+fresh])
        submit('finalize',['--partition=short','--time=00:15:00','--dependency=afterok:'+conf])
        all_jobs[system]=dict(campaign=str(out),jobs=jobs)
    path=runtime/(stamp+'_grid_audit_submissions.json');path.write_text(json.dumps(all_jobs,indent=2)+'\n')
    print('SUBMISSIONS='+str(path),flush=True)
if __name__=='__main__':main()
