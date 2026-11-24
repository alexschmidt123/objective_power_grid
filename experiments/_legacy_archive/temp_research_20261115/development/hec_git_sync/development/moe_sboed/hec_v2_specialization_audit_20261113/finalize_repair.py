"""Publish matched E4 refresh result after full completion/preservation checks."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v2_E4_refresh_20261113'
DEST=ROOT/'development/hec_git_sync/reports/hec_v2_E4_refresh_20261113'
def main():
    status=json.loads((RUN/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['historical_artifacts_preserved']
    result=json.loads((RUN/'RESULTS.json').read_text())
    for arm in ['ordinary','E4_refresh']:
        value=json.loads((RUN/arm/'RESULT.json').read_text())
        assert value['status']=='COMPLETE' and value['updates']==2000
        assert value['counts']==dict(actor=2000,critic=10000,posterior=300000)
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    (DEST/'SUMMARY.json').write_text(json.dumps(dict(result=result,artifact_path=str(RUN)),indent=2)+'\n')
    lines=['# Matched V2 E4 adapter refresh','',f"Accepted under locked joint competency/EIG rule: **{result['accepted']}**",'',f'Artifacts: `{RUN}`','']
    for k in ['competency','EIG']:lines.extend([f'## {k}','','```json',json.dumps(result[k],indent=2),'```',''])
    lines.extend(['## Limitations','',result['limitations'],'','Both arms:300 preparation steps and2000 unchanged native EIG updates. Control preparation updates only an unused diagnostic readout; candidate additionally updates adapter4. Additional adapter-backprop FLOPs are not exactly matched. Historical artifacts preserved.',''])
    (DEST/'RESULTS.md').write_text('\n'.join(lines));print(str(DEST))
if __name__=='__main__':main()
