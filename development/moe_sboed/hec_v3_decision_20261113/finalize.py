"""Publish compact V3 pilot results after completion/invariant checks."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v3_decision_20261113_pilot'
DEST=ROOT/'development/hec_git_sync/reports/hec_v3_decision_20261113'
def main():
    status=json.loads((RUN/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['historical_artifacts_preserved']
    result=json.loads((RUN/'RESULTS.json').read_text());phases={}
    for phase,updates in [('expert1',250),('expert2',250),('expert3',250),('selector',250),('control',1000)]:
        value=json.loads((RUN/phase/'RESULT.json').read_text());assert value['status']=='COMPLETE' and value['updates']==updates
        assert value['optimizer_steps']==dict(actor=updates,critic=5*updates,posterior=150*updates);phases[phase]=value
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    (DEST/'SUMMARY.json').write_text(json.dumps(dict(status=status,results=result,phases=phases,artifacts=str(RUN)),indent=2)+'\n')
    text=['# V3 decision-level MoE pilot','',f"Accepted under locked rule:**{result['accepted']}**",'',
       'Three complete V2-derived policies,independent action heads,categorical selector;uniform and standalone comparisons use identical physical execution. No role-specific specialization claim.','',
       '```json',json.dumps(result,indent=2),'```','',f'Artifacts:`{RUN}`','']
    (DEST/'RESULTS.md').write_text('\n'.join(text));print(str(DEST))
if __name__=='__main__':main()
