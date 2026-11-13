"""Validate and report a completed frozen-V2 confirmation; no training/inference."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=pathlib.Path,default=ROOT/'scratch_runs/hec_v2_confirmation_20261112')
    p.add_argument('--report-dir',type=pathlib.Path,default=ROOT/'development/hec_git_sync/reports/hec_v2_confirmation_20261112')
    p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args();run=a.run_dir.expanduser().resolve();report=a.report_dir.expanduser().resolve()
    if report.exists():raise FileExistsError('Preserve existing reports; no overwrite')
    cfg=json.loads((run/'CONFIG.json').read_text());c=json.loads((run/'COMPLETION.json').read_text())
    r=json.loads((run/'RESULTS.json').read_text())
    assert c['preserved'] and c['training_updates']==0
    if cfg['smoke']:
        assert a.allow_smoke and c['status']=='SMOKE_COMPLETE'
        assert 'scratch_runs' in report.parts,'Smoke reports must remain outside Git'
        outcome='INTEGRATION FIXTURE ONLY'
    else:
        assert c['status']=='COMPLETE' and cfg['primary_cases']==4096 and cfg['secondary_cases']==512
        assert cfg['levels']==[4000000,16000000]
        assert cfg['case_seed']==202611128001 and cfg['contrast_seed']==202611128002
        outcome='PRIMARY ROUTING ADVANTAGE ESTABLISHED' if r['primary_routing_superiority_established'] else 'PRIMARY ROUTING ADVANTAGE NOT ESTABLISHED'
    primary=r['primary_learned_minus_uniform'][str(cfg['levels'][-1])]
    assert bool(primary['CI95'][0]>0 and not cfg['smoke'])==r['primary_routing_superiority_established']
    for name in ['original','uniform','expert1','expert2','expert3','expert4','vsoed']:
        n=r['methods'][name][str(cfg['levels'][-1])]['n']
        assert n==(cfg['primary_cases'] if name in ['original','uniform'] else cfg['secondary_cases'])
    report.mkdir(parents=True)
    summary=dict(config=cfg,completion=c,results=r,outcome=outcome,run=str(run),
                 provenance={f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in
                             ['CONFIG.json','COMPLETION.json','RESULTS.json','PRE_RUN_MANIFEST.json','BANK_PROVENANCE.json']})
    (report/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    text='# Independent frozen-V2 routing confirmation\n\nOutcome: **'+outcome+'**.\n\n'
    text+=f"Completed in{c['seconds']:.1f}s;zero optimizer/training updates,protected historical inputs and source unchanged. Source commit `{cfg['source_commit']}`.\n\n"
    text+='## Primary paired comparison\n\n'
    text+=f"{cfg['primary_cases']} fresh independent prior/noise cases;shared independent{cfg['levels'][-1]:,}-negative prior bank;positive included once. Locked checkpoints and seeds;no interim sample-size or model selection. One primary two-sided paired95% t interval.\n\n"
    text+='| Contrasts | Learned V2 mean | Uniform V2 mean | Paired difference |95% CI |\n|---:|---:|---:|---:|---|\n'
    for level in map(str,cfg['levels']):
        x=r['methods']['original'][level];y=r['methods']['uniform'][level];d=r['primary_learned_minus_uniform'][level]
        text+=f"|{int(level):,}|{x['mean']:.6f}|{y['mean']:.6f}|{d['mean']:+.6f}|[{d['CI95'][0]:+.6f},{d['CI95'][1]:+.6f}]|\n"
    text+='\nThe smaller bank is a sensitivity diagnostic,not another primary success opportunity. These are finite-contrast lower-bound estimates,not exact EIG.\n\n## Secondary paired controls\n\n'
    text+=f"First{cfg['secondary_cases']} fresh cases,predeclared before outcomes. Compare each control with learned V2 on exactly the SAME subset;Holm5 family. These n differ from the primary table,so do not subtract their unpaired overall means.\n\n"
    text+='| Control | Control mean at largest bank | Learned−control |95% paired CI | Holm5 p |\n|---|---:|---:|---|---:|\n'
    for name,d in r['secondary_learned_minus'].items():
        m=r['methods'][name][str(cfg['levels'][-1])]['mean']
        text+=f"|{name}|{m:.6f}|{d['mean']:+.6f}|[{d['CI95'][0]:+.6f},{d['CI95'][1]:+.6f}]|{d['Holm5_p']:.6g}|\n"
    text+='''
## Scope and next scientific decision

No new routing method was trained here. This determines whether the original V2's learned gates have a reproducible advantage under a better-resolved independent estimator. A positive result supports that fixed trained lineage;it does not establish multi-seed robustness,all-baseline superiority or an ICML-ready contribution. A negative/inconclusive result remains negative/inconclusive;no fallback primary test or automatic retraining is launched.

One-hot expert pathways retain shared history/head/SG reference,not separately trained standalone expert policies. Existing vsOED is a historical frozen checkpoint and not compute/initialization matched. DAD,RL-sBOED and ScoreBED were not reevaluated. Model/pretraining costs and additional evaluator likelihood work must be disclosed. The original1M resolution study and failed replacement selector are preserved separately;they are not reclassified as new-model successes.

The all-consumer pathway knockout was validated separately,but these secondary controls are fixed one-hot pathways,not knockout results. Expert competency/conditional specialization still requires its own evidence. No historical checkpoint,dataset or large bank enters Git;only source and compact reports synchronize.
'''
    text+=f"\nRaw scientific output: `{run}`.\n"
    (report/'RESULTS.md').write_text(text)
    print('FINALIZED '+json.dumps(dict(outcome=outcome,report=str(report),run=str(run))))


if __name__=='__main__':main()
