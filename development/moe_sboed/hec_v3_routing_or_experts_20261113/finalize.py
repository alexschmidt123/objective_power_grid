"""Publish the two-category frozen V3 audit."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v3_routing_or_experts_20261113'
DEST=ROOT/'development/hec_git_sync/reports/hec_v3_routing_or_experts_20261113'
def main():
    status=json.loads((RUN/'STATUS.json').read_text());assert status['status'] in ['COMPLETE','INCONCLUSIVE_POSTERIOR_QUALITY']
    cfg=json.loads((RUN/'CONFIG.json').read_text());assert not cfg['smoke']
    result=json.loads((RUN/'RESULTS.json').read_text()) if (RUN/'RESULTS.json').exists() else None
    if result:assert status['historical_artifacts_preserved'] and result['training_steps']==0
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    record=dict(status=status,config=cfg,result=result,artifacts=str(RUN))
    (DEST/'SUMMARY.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=['# Frozen V3 diagnosis: routing quality or expert design','',f"Status:**{status['status']}**",'',
       'Only two categories:critic/selector/credit issues belong to routing quality;conditional expert complementarity/settings belong to expert design. Existing V3 experts differ by independent training seeds,not deliberately designed sBOED strategies.','']
    if result:
        for key in ['spce','native_return']:lines.extend([f'## {key}','','```json',json.dumps(result[key],indent=2),'```',''])
        lines.extend(['## Limitations','',result['limitations'],''])
    else:lines.extend(['Posterior quality failed locked thresholds. No routing-versus-expert conclusion is supported.',''])
    lines.extend([f'Artifacts:`{RUN}`','']);(DEST/'RESULTS.md').write_text('\n'.join(lines));print(str(DEST))
if __name__=='__main__':main()
