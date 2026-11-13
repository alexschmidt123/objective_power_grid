"""Publish compact results only after all native experiment checks pass."""
import json, pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
RUN=ROOT/'scratch_runs/hec_v2_router_only_20261112'
DEST=ROOT/'development/hec_git_sync/reports/hec_v2_router_only_20261112'

def main():
    status=json.loads((RUN/'STATUS.json').read_text())
    assert status['status']=='COMPLETE' and status['historical_artifacts_preserved']
    result=json.loads((RUN/'RESULTS.json').read_text())
    arms={k:json.loads((RUN/k/'RESULT.json').read_text()) for k in ['router_only','ordinary']}
    for value in arms.values():
        assert value['status']=='COMPLETE' and value['updates']==2000
        assert value['counts']==dict(actor=2000,critic=10000,posterior=300000)
        assert value['frozen_actor_parameters_preserved'] and value['actor_buffers_preserved'] and value['router_changed']
    DEST.mkdir(parents=True,exist_ok=True)
    paths=[DEST/'SUMMARY.json',DEST/'RESULTS.md']
    assert not any(p.exists() for p in paths),'Preserve existing published report'
    summary=dict(run_path=str(RUN),status=status,results=result,arms=arms)
    paths[0].write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    text=['# HEC-v2 frozen-head router refinement','',
          'One change: train only the existing coordinator and response gate while fixing all other actor parameters. Native critic/posterior updates and information objective are unchanged.',
          '',f'Artifacts: `{RUN}`','',
          'Each arm completed 2,000 updates from the same iteration-16001 runtime. Primary evaluation uses 2,048 fresh episodes and 16 million prior contrasts. Final checkpoints only; no test-based selection.','',
          'Measured paired comparisons (native evaluator result fields):','']
    for key,value in result.items():
        if key.startswith(('primary_','secondary_')): text.extend([f'## {key}','', '```json',json.dumps(value,indent=2),'```',''])
    text.extend(['## Limitations','',result['limitations'],'',
                 'A gain is supported only if the primary paired confidence interval excludes zero in the positive direction. A null or negative result rejects this intervention as an improvement on this lineage; it does not prove routing impossible.','',
                 'Frozen actor parameters and buffers were verified unchanged in both arms; historical source/checkpoint hashes were verified after evaluation.',''])
    paths[1].write_text('\n'.join(text))
    print(json.dumps(dict(status='REPORTS_CREATED',paths=[str(p) for p in paths])))

if __name__=='__main__':main()
