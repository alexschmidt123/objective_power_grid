"""Small reviewable source/validation reports and completed-pilot reports only."""
import argparse,json,pathlib
from v3_native import n,ROOT,WORK
def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    dest=WORK/'reports/hec_v3_feature_20261114';dest.mkdir(parents=True,exist_ok=True)
    val=ROOT/'scratch_runs/hec_v3_feature_20261114_validation01';smoke=ROOT/'scratch_runs/hec_v3_feature_20261114_native_smoke02'
    control=ROOT/'scratch_runs/hec_v3_feature_20261114_control_smoke01'
    r=json.loads((val/'RESULTS.json').read_text());f=json.loads((val/'FUNCTIONAL.json').read_text());s=json.loads((smoke/'RESULTS.json').read_text());cs=json.loads((control/'RESULTS.json').read_text())
    assert r['maximum_score_error']==0 and f['status']=='PASS' and s['counts']==cs['counts']==dict(actor=1,critic=5,posterior=150)
    retirement=json.loads((dest/'RETIREMENT.json').read_text());assert all(n.sha(p)==h for p,h in retirement['protected_checkpoints'].items())
    info=dict(initialization=r,functional=f,native_smoke=s,control_smoke=cs,
        initial_checkpoint_sha256=n.sha(val/'INITIAL_actor.pt'),active_source='development/moe_sboed/hec_v3_feature_20261114',
        retired_sources='development/moe_sboed/retired/hec_v3_20261114',legacy_reference_loading='Verified archived hashes and loaded old actor checkpoint;finite2D forward passed')
    (dest/'VALIDATION.json').write_text(json.dumps(info,indent=2)+'\n')
    settings=(pathlib.Path(__file__).parent/'SETTINGS.md').read_text()
    (dest/'IMPLEMENTATION.md').write_text(settings+'\n\n## Measured validation\n\n'+
        f"The complete original V2 score vector was reproduced exactly over2,000 episodes and1M contrasts: mean {r['mean']:.8f} nats, maximum score/action errors0. This is initialization preservation,not an improvement.\n\n"+
        f"Registered parameters: {f['parameters']:,}; trainable innovation/router parameters: {f['trainable_parameters']:,}. Fixed provider weights stored as buffers are additional. Standalone execution, prefix masking, finite gradients and hidden-layer connectivity passed. Candidate and control each passed one native update with1 actor/5 critic/150 posterior steps.\n\n"+
        'The failed V3 active source directories now contain retirement notices; immutable source and checkpoints remain available for historical comparisons. No old campaign was restarted.\n')
    if not args.pilot:return
    out=ROOT/'scratch_runs/hec_v3_feature_20261114_pilot';status=json.loads((out/'STATUS.json').read_text())
    if status['status']!='COMPLETE':
        (dest/'RUN_STATUS.md').write_text('# Replacement V3 pilot status\n\n'+json.dumps(status,indent=2)+'\n\nRaw artifacts: '+str(out)+'\n');return
    result=json.loads((out/'RESULTS.json').read_text());assert result['protocol']==r['protocol']
    assert all(n.sha(path)==h for path,h in json.loads((out/'MANIFEST.json').read_text()).items())
    lines=['# Replacement V3 pilot results','',f"Accepted against all predeclared comparators: **{result['accepted']}**.",'',
        'Exact original V2 evaluation:2,000 cases/1M contrasts,frozen scorer and physics. No comparison uses the previous16M bank.','',
        '| Policy | Estimated EIG(nats) |','|---|---:|']
    for k,v in result['means'].items():lines.append(f'| {k} | {v:.8f} |')
    lines+=['','| Learned minus comparator | Difference | Paired95% CI | Holm p |','|---|---:|---|---:|']
    for k,v in result['comparisons'].items():lines.append(f"| {k} | {v['mean']:+.8f} | [{v['CI95'][0]:+.8f},{v['CI95'][1]:+.8f}] | {v['p_Holm']:.6g} |")
    lines+=['',result['limitations'],'',result['reference_reuse'],'',f'Raw artifacts: `{out}`.','']
    (dest/'RESULTS.md').write_text('\n'.join(lines));(dest/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
