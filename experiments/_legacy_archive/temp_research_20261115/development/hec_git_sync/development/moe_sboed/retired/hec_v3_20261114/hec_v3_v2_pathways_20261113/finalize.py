import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v3_v2_pathways_20261113_pilot'
DEST=ROOT/'development/hec_git_sync/reports/hec_v3_v2_pathways_20261113'
def main():
    status=json.loads((RUN/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['historical_artifacts_preserved']
    result=json.loads((RUN/'RESULTS.json').read_text())
    for name,updates in [(f'expert{k}',250) for k in range(1,5)]+[('selector',250),('control',1250)]:
        item=json.loads((RUN/name/'RESULT.json').read_text());assert item['status']=='COMPLETE' and item['updates']==updates
        assert item['optimizer_steps']==dict(actor=updates,critic=5*updates,posterior=150*updates)
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    summary=dict(status=status,results=result,distillation=json.loads((RUN/'DISTILLATION.json').read_text()),artifacts=str(RUN))
    (DEST/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# V2-based four-pathway V3 pilot','',f"Accepted under locked all-baseline rule:**{result['accepted']}**",'',
       'Four inherited V2 query computations,each with own bounded head;original commander/coordinator retained. No three-copy design. No averaging of experimental locations.','',
       '```json',json.dumps(result,indent=2),'```','',f'Artifacts:`{RUN}`','']
    (DEST/'RESULTS.md').write_text('\n'.join(lines));print(str(DEST))
if __name__=='__main__':main()
