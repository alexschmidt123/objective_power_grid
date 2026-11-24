"""Write compact reports for completed or posterior-quality-inconclusive run."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v2_expert_contribution_20261113_posterior_refined'
DEST=ROOT/'development/hec_git_sync/reports/hec_v2_expert_contribution_20261113_posterior_refined'
def main():
    status=json.loads((RUN/'STATUS.json').read_text())
    assert status['status'] in ['COMPLETE','INCONCLUSIVE_POSTERIOR_QUALITY']
    cfg=json.loads((RUN/'CONFIG.json').read_text());assert not cfg['smoke']
    result=json.loads((RUN/'RESULTS.json').read_text()) if (RUN/'RESULTS.json').exists() else None
    if status['status']=='COMPLETE':assert status['historical_artifacts_preserved'] and status['optimizer_steps']==0
    quality=json.loads((RUN/'POSTERIOR_QUALITY.json').read_text())
    DEST.mkdir(parents=True,exist_ok=True)
    assert not (DEST/'SUMMARY.json').exists() and not (DEST/'RESULTS.md').exists()
    record=dict(status=status,configuration=cfg,result=result,posterior_quality=quality,artifact_path=str(RUN))
    (DEST/'SUMMARY.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    text=['# Frozen V2 expert contribution diagnostic','',f"Status: **{status['status']}**",'',f'Artifacts: `{RUN}`','',
          'No training. Single-decision and persistent all-consumer expert removals. History regimes are fitted on selection histories and tested on independent histories and simulation banks.','']
    if result:
        for k,v in result.items():
            if k.startswith(('primary_','secondary_')) or k in ['expert_removal_effects','regime','test_regimes']:text.extend([f'## {k}','','```json',json.dumps(v,indent=2),'```',''])
        text.extend(['## Limitations','',result['limitations'],''])
    else:text.extend(['Posterior sampling quality failed the locked ESS/maximum-weight criteria. No EIG/routing conclusion is supported; observed histories were not replaced.',''])
    text.extend(['Posterior samples are finite importance approximations. ESS does not establish complete mode coverage. Finite prior-sPCE is not exact EIG. No deployable history selector or repeated-routing improvement was trained or tested.',''])
    (DEST/'RESULTS.md').write_text('\n'.join(text))
    print(json.dumps(dict(status='REPORTS_CREATED',path=str(DEST))))
if __name__=='__main__':main()
