"""Publish completed audit and evidence-selected next intervention."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v2_specialization_audit_20261113'
DEST=ROOT/'development/hec_git_sync/reports/hec_v2_specialization_audit_20261113'
def main():
    status=json.loads((RUN/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['historical_artifacts_preserved']
    result=json.loads((RUN/'RESULTS.json').read_text());noise=json.loads((RUN/'LABEL_NOISE.json').read_text());assert not result['smoke']
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    (DEST/'SUMMARY.json').write_text(json.dumps(dict(audit=result,label_noise=noise,artifact_path=str(RUN)),indent=2)+'\n')
    lines=['# Fresh corrected V2 specialization audit','',
       'Confirmed diagnostic issue: old evaluate_v2.py:89 removed response coordinates0:12 from every expert, dropping learned E1/E3 coordinates. Corrected128-coordinate probe masks only E2/E4 analytic prefixes. Frozen actor code/checkpoints remain unchanged.','',
       '640 fresh episodes:256 calibration,128 rival selection,256 independent test. Targets are existing model-derived tasks,not physical ground truth or exact EIG.','',
       '| Expert | Relative MSE reduction vs constant | Designated advantage95%CI |','|---|---:|---|']
    for k,v in result['experts'].items():lines.append(f"|{k}|{100*v['relative_MSE_reduction']:.2f}%|{v['designated_advantage']['CI95']}|")
    lines.extend(['','All expert packets have nonzero direct-fusion action sensitivity. Old E2/E4 diagnostic readouts underperform constants, but those readouts are absent from the deployed action computation.','',
       'E4 probe MSE0.198640 versus estimated target MC noise0.039044; excess0.159596,95%CI[0.114027,0.205165]. This supports testing an E4 adapter refresh using its existing task; it does not establish a network bug.','',
       'Chosen next intervention: candidate refreshes existing adapter4 plus its diagnostic head; control refreshes only the unused diagnostic head. Both get matched preparation data/steps and2000 native EIG updates. Acceptance requires independent fresh competency and EIG gains. Full locked protocol is development/moe_sboed/hec_v2_specialization_audit_20261113/REPAIR_PROTOCOL.md.','',
       result['limitations'],''])
    (DEST/'RESULTS.md').write_text('\n'.join(lines));print(str(DEST))
if __name__=='__main__':main()
