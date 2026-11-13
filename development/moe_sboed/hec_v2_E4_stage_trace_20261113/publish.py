"""Publish the completed trace without changing historical artifacts."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v2_E4_stage_trace_20261113'
DEST=ROOT/'development/hec_git_sync/reports/hec_v2_E4_stage_trace_20261113'
def main():
    status=json.loads((RUN/'STATUS.json').read_text());result=json.loads((RUN/'RESULTS.json').read_text())
    assert status['status']=='COMPLETE' and status['historical_artifacts_preserved']
    assert result['historical_prepared_state_digests_match'] and result['historical_final_evaluation_reproduced']
    traces={arm:json.loads((RUN/f'{arm}_PREPARATION_TRACE.json').read_text()) for arm in ['ordinary','E4_refresh']}
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    summary=dict(result=result,preparation_traces=traces,artifact_path=str(RUN))
    (DEST/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# E4 preparation stage trace','',
       '**Finding: the short preparation never demonstrated improved heldout representation competency. The experiment does not support freezing the refreshed adapter to preserve a supposed gain.**','',
       'Prepared checkpoints were absent from the old run. Both short preparation phases were reproduced, saved separately and verified against their exact historical state digests and receipts. Original/final episode-level evaluations also reproduce. Zero new native training updates.','',
       '| Stage |128-coordinate probe MSE |52-coordinate adapter probe MSE |Diagnostic readout MSE |','|---|---:|---:|---:|']
    for key,value in result['stages'].items():lines.append(f"|{key}|{value['probe128_MSE']:.9f}|{value['adapter52_probe_MSE']:.9f}|{value['existing_readout_MSE']:.9f}|")
    lines.extend(['','## Representation comparisons','','```json',json.dumps(result['competency'],indent=2),'```','',
       '## Optimization observations','',
       'Candidate full-calibration readout MSE:0.272795 after ridge calibration,0.820647 after first Adam step,0.286441 after300steps. This records a first-step overshoot and no net full-calibration readout improvement over initialization; it is not proof of an incorrect gradient or intrinsic expert incapacity.','',
       'Preparation changes adapter weights by4.99% relativeL2 and common-history actions byRMS0.0003573. Thus updates reach the deployed policy, but the measured target representation does not improve reliably. The unused readout improves through calibration; control head-only preparation achieves lower heldout readout error0.163438 than adapter+head0.179851. This is not representation specialization.','',
       '## Decision','',
       'Do not launch the frozen-refreshed-E4 native-training experiment. First establish a heldout representation improvement in a bounded preparation-only optimization/target-learnability study. These results locate the failed intervention before native continuation; they do not rule out other specialization methods.','',
       result['limitations'],'',f'Artifacts: `{RUN}`',''])
    (DEST/'RESULTS.md').write_text('\n'.join(lines));print(str(DEST))
if __name__=='__main__':main()
