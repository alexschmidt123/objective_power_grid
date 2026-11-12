# Router and commander gradient audit

Audit date: 2026-11-12. Exact object: frozen V2Actor candidate, native iteration16001, actor SHA256 `90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108`. This is not V2.1, RR-MoE, or the later routing pilot.

Source aliases on LabPC: `B=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109`; `A=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_internal_ablation`; `V=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101/source/src/research/vendor/vsoed`. Locations below use these exact aliases and line numbers. Measured CPU evidence: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_code_audit/verification_02/MEASURED_AUDIT.json`. Five deterministic unit tests plus actual-checkpoint tests; 114 protected files hash-preserved, zero optimizer steps, zero new observation simulations, zero GPU jobs. No native objective or historical model/source was changed.

## Exact router and command code

Commander context=[live history128,SG provisional design/4 two,belief mean4,log1p variance4,component-mean correlation6]=144. Commander144→64 Tanh→20; add stage_logits[t]−reference_logits[t], then Tanh. Split4/6/6/4 drives actual region attention, likelihood probes/hypothesis tilts, Fisher location/direction and prospective-information queries. Commands are not routing weights.

Coordinator input=[history128,legacy expert features4×64]=384. `coordinator`:384→64 Tanh→4 (24,900 parameters). `response_gate`:256→4,no bias (1,024). `raw=coordinator(context)+response_gate(flatten(responses))`; `g=softmax(10*tanh(raw/10),dim=-1)`. This produces four positive continuous weights summing to1, not expert execution probabilities. Stage_logits has600 parameters; commander10,580. Both expert channels receive the same g.

Initialization: coordinator inherited trained HEC-v1 weights; it originally had a zero output layer at HEC-v1 creation. New V2 response gate is zero initialized. The existing actor Adam group is expanded/replaced with model.parameters while retaining old compatible objects/moments; new parameters start native Adam fresh. No optimizer omission was detected.

| Module | Native-loss gradient L2 | Weight change L2 | Saved Adam steps | All parameters in optimizer |
|---|---:|---:|---:|---|
|commander|0.041221389|2.7012577|4000–4000|True|
|coordinator|0.011426633|4.6388563|4000–4000|True|
|response_gate|0.012041321|0.64008313|2000–2000|True|

All tested gradients are finite and nonzero. Multi-chunk checkpoint recomputation (chunk7 vs direct path) relative gradient error **2.09e-07**. This tests the real checkpoint closure and chunk binding, not a toy router. Float64 raw-logit→action finite-difference max error **1.3e-11**; cached float32 expert features and actual head weights were promoted only for this numerical check.

## Variation and sensitivity

| Expert | Mean gate over512×30 histories | Mean within-stage SD |
|---|---:|---:|
|E1|0.126743|0.040406|
|E2|0.293661|0.060772|
|E3|0.119664|0.052105|
|E4|0.459932|0.073017|

Variation is measured, but not used as proof of good routing.

| t (zero based) | Raw squash derivative mean | Gate-to-action Jacobian RMS | Learned-vs-uniform action RMS |
|---:|---:|---:|---:|
|0|0.999304|0.012638|0.008227|
|3|0.994758|0.041744|0.033057|
|12|0.991578|0.025710|0.070058|
|24|0.990384|0.024442|0.032699|
|29|0.986775|0.029595|0.033061|

The tanh-logit transform is not saturated on these40 states, router gradients do not vanish, and changing gates changes actions. These are local sensitivity checks, not evidence that those changes improve EIG. The actual actor objective is−mean live critic Q, not direct differentiation of independently measured full-trajectory EIG; critic ranking/calibration as a cause of routing weakness remains unverified.

Locations: `B/source/v2_policy.py:19`, `B/source/v2_policy.py:28`, `B/source/v2_policy.py:30`, `B/source/migrate_v2.py:19`, `V/pg_vsoed.py:697–720` (native actor loss and update). No native update is executed in this audit.
