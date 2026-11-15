"""MacBook monitor: report/sync completed pilot; never starts or restarts training."""
import argparse,json,pathlib,subprocess,time
LOCAL=pathlib.Path.home()/'scratch_runs/hec_v3_feature_20261114'
REMOTE='~/Documents/objective_power_grid_temp';WORK=REMOTE+'/development/hec_git_sync'
RUN=REMOTE+'/scratch_runs/hec_v3_feature_20261114_pilot'
def remote(command):return subprocess.run(['ssh','labpc',command],text=True,capture_output=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    while True:
        r=remote('cat '+RUN+'/STATUS.json')
        if r.returncode:status=dict(status='NO_STATUS_YET',stderr=r.stderr[-1000:])
        else:status=json.loads(r.stdout)
        print(json.dumps(status),flush=True);(LOCAL/'WATCH_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        if args.once:return
        if status['status'] in ['COMPLETE','FAILED','INCOMPLETE']:
            published=remote('/home/grads/g/g.lin/miniconda3/envs/dellma/bin/python '+WORK+'/development/moe_sboed/hec_v3_feature_20261114/publish.py --pilot')
            if published.returncode:raise RuntimeError(published.stdout+published.stderr)
            cmd=['python3',str(pathlib.Path.home()/'scratch_runs/moe_workflow_sync_20261109/moe_sync.py'),
                '--message','Record replacement V3 native-EIG pilot outcome','--files','reports/hec_v3_feature_20261114',
                '--receipt',str(LOCAL/'PILOT_VERIFIED_SYNC.json')]
            subprocess.run(cmd+['--review-only'],check=True)
            patch=remote('git -C '+WORK+' diff --cached -- reports/hec_v3_feature_20261114')
            assert patch.returncode==0;(LOCAL/'PILOT_REVIEW.diff').write_text(patch.stdout)
            subprocess.run(cmd,check=True)
            target=LOCAL/'pilot_report';target.mkdir(exist_ok=True)
            files=['RESULTS.md','SUMMARY.json'] if status['status']=='COMPLETE' else ['RUN_STATUS.md']
            subprocess.run(['scp',*['labpc:'+WORK+'/reports/hec_v3_feature_20261114/'+name for name in files],str(target)],check=True)
            return
        time.sleep(30)
if __name__=='__main__':main()
