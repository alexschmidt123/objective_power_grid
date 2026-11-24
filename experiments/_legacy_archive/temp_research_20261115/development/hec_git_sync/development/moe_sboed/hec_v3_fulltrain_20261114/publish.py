"""Publish lightweight training receipts/results, never checkpoints or datasets."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
RUN=ROOT/'scratch_runs/hec_v3_fulltrain_20261114_run01';OUT=WORK/'reports/hec_v3_fulltrain_20261114'
def main():
    OUT.mkdir(exist_ok=True,parents=True);status=json.loads((RUN/'STATUS.json').read_text())
    summary=dict(status=status,run=str(RUN),launch=json.loads((ROOT/'scratch_runs/hec_v3_fulltrain_20261114_run01.LAUNCH.json').read_text()),configuration=json.loads((RUN/'CONFIG.json').read_text()))
    for name in ['candidate_warmup','candidate_joint']:
        p=RUN/name/'RESULT.json'
        if p.exists():summary[name]=json.loads(p.read_text())
    p=RUN/'RESULTS.json'
    if p.exists():summary['evaluation']=json.loads(p.read_text())
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    text='# Full-training V3 continuation\n\nStatus: '+status['status']+'.\n\nV2 and other frozen baselines reused; no baseline retraining. V3 starts at original terminal V2, with 200 history-frozen plus 1800 history-trainable native updates. Native objective unchanged. One training lineage; uniform/singles are inference-only ablations. Historical baselines are not end-to-end compute matched.\n\n'
    if 'evaluation' in summary:
        r=summary['evaluation'];text+='| Policy | EIG (nats) |\n|---|---:|\n'+''.join('| '+k+' | '+format(v,'.8f')+' |\n' for k,v in r['means'].items())
        text+='\nAccepted against predeclared comparisons: '+str(r['accepted'])+'.\n\n| Learned minus | Difference | Paired 95% CI |\n|---|---:|---|\n'
        for k,v in r['comparisons'].items():text+='| '+k+' | '+format(v['mean'],'+.8f')+' | '+str(v.get('ci95',v.get('ci_95',v)))+' |\n'
        text+='\nCase-based paired uncertainty is conditional on this training lineage. See SUMMARY.json for multiplicity correction and protocol certificates.\n'
    text+='\nRaw artifacts: `'+str(RUN)+'`.\n';(OUT/'RESULTS.md').write_text(text)
    print(json.dumps(dict(report=str(OUT),status=status['status'])))
if __name__=='__main__':main()
