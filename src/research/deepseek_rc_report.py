
"""Evidence-based reporting; all analyses remain on LabPC."""
import json,time,copy,csv
from pathlib import Path
import numpy as np,torch
from src.research.deepseek_sboed_experiment import write,sha
from src.research.deepseek_sboed_v2_experiment import paired
from src.research.journal_validation import CommonValidation

def diagnostics(C,cfg,m):
 from src.research.deepseek_rc_competence import competence,feature_audit,plots
 begin=time.monotonic();D=C/'diagnostics'
 for name in ['data','costs','figures']:(D/name).mkdir(parents=True,exist_ok=True)
 with torch.no_grad(),torch.device('cuda'):
  feature_audit(m,cfg,D);competence(m,cfg,D,begin+600)
 torch.cuda.synchronize()
 write(D/'costs/total.json',dict(exclusive_GPU_wall_seconds=time.monotonic()-begin,soft_budget_seconds=600,optimizer_updates=0,exact_kernel_busy_seconds=None))
 plots(D)

def finish_report(C,cfg):
 official=json.loads((C/'reports/RC_seed1001_result.json').read_text())
 reference=Path.home()/'Documents/objective_power_grid_ir_analysis/reports/ir_moe_labpc_diagnostics'
 provenance=json.loads((reference/'source_provenance.json').read_text())
 assert provenance['canonical_cases_sha256']==cfg['cases_sha256']
 assert provenance['canonical_contrast_sha256']==cfg['contrast_sha256']
 assert provenance['paired_draws_sha256']==official['metadata']['paired_draws_sha256']
 with (reference/'paired_trajectory_scores.csv').open() as f:rows=list(csv.DictReader(f))
 assert [int(x['trajectory_id']) for x in rows]==list(range(2000))
 v=np.load(C/'evaluations/RC_seed1001_scores.npy')
 comparisons={name:dict(reference_EIG=float(np.mean([float(r[name]) for r in rows])),RC_minus_reference=paired(v,[float(r[name]) for r in rows])) for name in ['IR_MoE','vsOED','DeepSeekMoE_v1','corrected_DAD']}
 write(C/'reports/paired_all_baselines.json',dict(comparisons=comparisons,provenance=provenance,source_CSV_sha256=sha(reference/'paired_trajectory_scores.csv')))
 confirmation=json.loads((C/'reports'/f"RC_seed{cfg['confirmation_seed']}_result.json").read_text())
 cost=json.loads((C/'costs/full_final.json').read_text());progress=json.loads((C/'costs/full_progress.json').read_text())['records']
 dev=json.loads((C/'reports/development_comparisons.json').read_text());lat=json.loads((C/'reports/latency.json').read_text())
 feat=json.loads((C/'reports/final_contribution.json').read_text())['stages'];smoke=json.loads((C/'readiness/GPU_smoke.json').read_text())
 cp=json.loads((C/'diagnostics/data/competence.json').read_text())
 reliable=[r for r in cp['histories'] if 'paired_advantage_over_selected' in r]
 wins=sum(sum(r['reliable_alternative_wins']) for r in reliable)
 learned_uniform=paired(np.load(C/'evaluations/development_full.npy'),np.load(C/'evaluations/development_uniform_subset.npy'))
 learned_weights=paired(np.load(C/'evaluations/development_full.npy'),np.load(C/'evaluations/development_uniform_weights.npy'))
 doc=f"""# DeepSeekMoE-sBOED-RC seed101 results

Official EIG: **{official['EIG']:.8f} +/- {official['MC_SE']:.8f} nats** (trajectory MC SE).
Protocol: seed1001,2000 trajectories,1000000 contrasts,H30; exact canonical banks.
Independent confirmation seed{cfg['confirmation_seed']}: {confirmation['EIG']:.8f} +/- {confirmation['MC_SE']:.8f}.
One training seed; these errors do not measure across-training-seed uncertainty.

| Baseline | Reference EIG | RC minus reference | Paired MC SE |95% CI |
|---|---:|---:|---:|---|
"""
 for k,x in comparisons.items():
  p=x['RC_minus_reference'];doc+=f"|{k}|{x['reference_EIG']:.8f}|{p['difference']:.8f}|{p['paired_MC_SE']:.8f}|{p['interval95']}|\n"
 doc+=f"""
Pairing verified by canonical truth/noise/bank identity, all2000 trajectory indices, and the historical audited raw-score export. Source provenance and export SHA are in paired_all_baselines.json.

Checkpoint selection: latest compute-eligible checkpoint, not official EIG or best test score. No retuning after testing.
Executed updates: {cost['executed_updates']}; selected update: {cost['selected_update']}.
Training cost (full ledger follows): {json.dumps(cost,indent=2)}.

Latency: {json.dumps(lat)}. Only a frozen forward pass is deployed. No simulation, teacher, candidate design search or online optimization occurs.

Main preclip >5 fraction among logged updates: {np.mean([x['preclip_gradient_norm']>5 for x in progress]):.6f}.
Final gain {feat[-1]['gain']:.8f}; mean shared feature norm {np.mean([r['shared_norm'] for r in feat]):.8f}; mean calibrated routed norm {np.mean([r['calibrated_routed_norm'] for r in feat]):.8f}.
Learned minus random-uniform-subset development routing: {json.dumps(learned_uniform)}.
Learned minus uniform weights on unchanged selected sets: {json.dumps(learned_weights)}.
Development and ablation results: {json.dumps(dev,indent=2)}.

## Specialization

NOT YET ESTABLISHED. The conditional diagnostics have {len(reliable)} usable finite importance references out of {len(cp['histories'])}, and {wins} familywise screened alternative-set wins. ESS screening is necessary but not a proof of posterior convergence. Low-ESS rows, especially late stages, are withheld. Paired fantasy SE conditions on finite posterior references; independent-bank sensitivity is retained. Larger features and changed designs alone are not evidence of information specialization. Any contextual win must also be reproducible and exploited by the router with full-horizon benefit. See diagnostics/data/competence.json and competence_matrix.png.

No architecture superiority claim is justified by one seed. Dense controls have slightly different parameter counts and throughput, and frozen masking interventions are not retrained controls.
"""
 for path in [C/'reports/DEEPSEEK_RC_SEED101_RESULTS.md',Path('reports/DEEPSEEK_RC_SEED101_RESULTS.md')]:path.write_text(doc)
 contribution=f"""# Routed contribution and specialization

The historical IR limitation was weak routed influence, not a proven gate-normalization defect. Original IR shared/routed feature norms were0.186664/0.029860 and shared/routed action sensitivity differed by6.01x. This is a different trained checkpoint and routing objective; changes relative to it are descriptive, not a controlled causal comparison.

RC final mean local routed-removal design RMS: {np.mean([r['mask_routed_design_RMS'] for r in feat]):.8f}; shared-removal RMS: {np.mean([r['mask_shared_design_RMS'] for r in feat]):.8f}.
Mean controlled-swap design RMS: {np.mean([r['swap_design_RMS'] for r in feat]):.8f}.
Actual sequential usefulness is assessed by paired development interventions, not these feature magnitudes: {json.dumps(dev,indent=2)}.

Learned-minus-uniform subsets: {json.dumps(learned_uniform)}.
Learned-minus-uniform selected weights: {json.dumps(learned_weights)}.

Conditional information competence uses all35 routed triples, two independent262144-prior-particle references, ESS>=128 on both,32 paired fantasies per reference,65536 posterior contrasts, same deterministic continuation, and familywise screening. Truth that generated the prefix is discarded. Immediate and remaining-horizon information are recorded separately. All unreliable middle/late rows are excluded. Reliable histories {len(reliable)}/{len(cp['histories'])}; screened alternative wins {wins}. Useful specialization remains NOT YET ESTABLISHED pending all required contextual and routing-effect criteria.

![Utilization](../experiments/10252026/10252026_deepseek_rc_seed101/diagnostics/figures/expert_utilization.png)
![Competence](../experiments/10252026/10252026_deepseek_rc_seed101/diagnostics/figures/competence_matrix.png)

Detailed pre/postclip gradients, gain, seven expert gradient norms, entropy and utilization are in costs/full_progress.json. Smoke computation: {json.dumps(smoke)}.
"""
 for path in [C/'reports/DEEPSEEK_RC_EXPERT_CONTRIBUTION.md',Path('reports/DEEPSEEK_RC_EXPERT_CONTRIBUTION.md')]:path.write_text(contribution)
