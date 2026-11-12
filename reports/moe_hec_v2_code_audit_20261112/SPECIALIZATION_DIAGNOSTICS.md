# Specialization diagnostics

Audit date: 2026-11-12. Exact object: frozen V2Actor candidate, native iteration16001, actor SHA256 `90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108`. This is not V2.1, RR-MoE, or the later routing pilot.

Source aliases on LabPC: `B=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109`; `A=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_internal_ablation`; `V=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101/source/src/research/vendor/vsoed`. Locations below use these exact aliases and line numbers. Measured CPU evidence: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_code_audit/verification_02/MEASURED_AUDIT.json`. Five deterministic unit tests plus actual-checkpoint tests; 114 protected files hash-preserved, zero optimizer steps, zero new observation simulations, zero GPU jobs. No native objective or historical model/source was changed.

## Distinct computation is verified; useful four-way specialization is not

A+.05 perturbation within each command block changes its own expert packet, with exactly zero packet change in the other three pathways on the tested histories. Expert MLPs consume distinct model-derived inputs, have different widths, have finite gradients, and changed during historical optimization. These facts reject the simple explanation “four untouched copies computing the same thing.” They do not prove four complementary decision strategies.

## Same-history feature similarity

Linear centered kernel alignment(CKA), on512 trajectories×stages3,12,24=1,536 saved states. CKA is invariant to orthogonal feature-basis changes; direct coordinate cosine between independently learned packet spaces would be misleading. Global CKA includes stage variation; per-stage matrices are also saved in measured evidence. Zero-variance features yield undefined rather than a false similarity score.

Full128-dimensional packet CKA:

| | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
|E1|1.00000|0.43465|0.04351|0.51562|
|E2|0.43465|1.00000|0.15305|0.41159|
|E3|0.04351|0.15305|1.00000|0.07077|
|E4|0.51562|0.41159|0.07077|1.00000|

Learned116-coordinate probe representation CKA (first64+response tail52):

| | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
|E1|1.00000|0.32368|0.04325|0.19713|
|E2|0.32368|1.00000|0.15019|0.24678|
|E3|0.04325|0.15019|1.00000|0.14223|
|E4|0.19713|0.24678|0.14223|1.00000|

The12 analytic coordinates are excluded for E2/E4 in the validated competency probe. The equal116-dimensional convention also drops12 learned response coordinates for E1/E3; this is a probe-design limitation, not a complete assay of everything encoded by those experts. The `features=first+response` diagnostic field is NOT used as a substitute for the true128-dimensional packet here.

Same-basis cosine of actual consumed hidden contributions U_k, averaged on the same1,536 states:

| | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
|E1|1.00000|0.19840|0.11874|0.16371|
|E2|0.19840|1.00000|0.25746|0.23535|
|E3|0.11874|0.25746|1.00000|0.29687|
|E4|0.16371|0.23535|0.29687|1.00000|

These are not near-identical contributions. Neither low CKA nor distinct consumed directions proves useful conditional information advantage.

## Existing validated task prediction, independently recomputed

Ridge probes calibrated on256 histories×3 stages; held-out256 histories; standardization and rival selection use calibration only. Standardized MSE is appropriate here, not classification accuracy. Rows=expert representation, columns=geometry/discrimination/Fisher/opportunity task; lower is better.

| | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
|E1|0.23005|0.50315|0.75589|0.40908|
|E2|0.22126|0.23964|0.56999|0.40895|
|E3|0.64469|0.62764|0.27076|0.40382|
|E4|0.56150|0.29127|0.56661|0.41132|

Designated advantage = held-out loss of the calibration-selected strongest rival minus designated expert loss, averaged within each validation history over3 stages. The saved raw loss arrays reproduce the reported contrasts to1e-12.

| Expert | Designated advantage | 95% CI | Holm4 p |
|---|---:|---|---:|
|expert1|-0.008795|[-0.031577,+0.013986]|0.895563|
|expert2|+0.051629|[+0.029018,+0.074241]|3.14806e-05|
|expert3|+0.295845|[+0.255773,+0.335917]|1.09476e-34|
|expert4|-0.002370|[-0.032075,+0.027336]|0.895563|

**E2 and E3 pass designated-task advantages; E1 and E4 do not.** Lack of significance is not proof that E1/E4 contain no useful information. Frozen preparation readouts also fail post-training constant-predictor comparisons:

| Frozen readout | Constant MSE minus readout MSE | Holm2 p |
|---|---:|---:|
|expert2|-2.218735|2.60779e-110|
|expert4|-0.532759|7.8053e-72|

Negative differences mean the inherited frozen readout is worse than the calibration constant predictor. This supports a lack of retained readout alignment after native training; it does not prove representation forgetting, because a freshly calibrated probe can recover E2's task information. Experts continue changing while readouts are frozen and auxiliary loss=0, by design. The actual amount caused by this schedule is unestablished without a controlled comparison.

## Gradient similarity in a comparable space

Comparing raw parameter gradients across different-sized expert networks is undefined. Instead, isolate each expert's contribution to the actual native critic actor loss and differentiate with respect to the SAME128-dimensional history embedding; detach gates, direct history path and the other expert contributions. Concatenate40 case gradients and take cosine:

| | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
|E1|1.00000|-0.02916|0.09675|-0.04508|
|E2|-0.02916|1.00000|0.05147|-0.06504|
|E3|0.09675|0.05147|1.00000|-0.14203|
|E4|-0.04508|-0.06504|-0.14203|1.00000|

The local gradient directions differ and are finite; this is evidence against identical local computations, not conditional EIG specialization.

## Existing conditional contributions and regime interactions

Recomputed24 saved conditional tests:128 joint-generated prefixes,4 future-noise replicates,stages3/12/24,calibration-median uncertainty regimes. Preserve true latent paired with its observed prefix, change one donor expert packet at the next decision, resimulate future observations under changed actions, then continue V2 normally. Replicates are clustered by history. The score is remaining normalized native variational return, using the restored frozen HEC-v1 posterior at14001; it is NOT independently contrast-scored conditional EIG or exact conditional MI. Gates are computed before donor replacement.

Holm24 significant individual contribution tests: **0/24**. All raw effect means reproduce saved JSON to1e-12. An explicitly exploratory high-minus-low regime interaction analysis on the existing per-case mean effects also finds **0/12** Holm12-significant interactions. This additional family is an audit diagnostic, not a predeclared new EIG result.

The full model beats each frozen one-hot pathway in the2,000-case EIG study, supporting collaboration within this checkpoint. It does not establish each expert's unique conditional advantage, individual standalone quality, or expert chronology. One-hot/donor interventions can shift feature scales/distributions.

Locations: `B/source/evaluate_v2.py:92`, `B/source/evaluate_v2.py:102`, `B/source/evaluate_v2.py:103`, `B/source/conditional_diagnostics.py:20`, `B/source/conditional_diagnostics.py:28`. New CKA/gradient diagnostics generate no new observations or training data.
