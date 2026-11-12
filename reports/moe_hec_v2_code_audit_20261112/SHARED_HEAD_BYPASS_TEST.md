# Shared-head bypass, sensitivity and removal semantics

Audit date: 2026-11-12. Exact object: frozen V2Actor candidate, native iteration16001, actor SHA256 `90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108`. This is not V2.1, RR-MoE, or the later routing pilot.

Source aliases on LabPC: `B=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109`; `A=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_internal_ablation`; `V=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101/source/src/research/vendor/vsoed`. Locations below use these exact aliases and line numbers. Measured CPU evidence: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_code_audit/verification_02/MEASURED_AUDIT.json`. Five deterministic unit tests plus actual-checkpoint tests; 114 protected files hash-preserved, zero optimizer steps, zero new observation simulations, zero GPU jobs. No native objective or historical model/source was changed.

## Exact graph decomposition

Let first_k,response_k each be64-vectors. The128-dimensional head preactivation is

`pre = b + W_history*s + sum_k g_k*(W_k*first_k + R_k*response_k)`.

Then delta=tanh(W_out*ReLU(pre)+b_out) and action=a0+4delta−a0*abs(delta). Define U_k=W_k*first_k+R_k*response_k. Every U_k occupies the same consumed hidden basis; raw expert packet coordinates do not.

**A. Confirmed structural alternative paths, not a coding error.** History enters the fusion directly, and frozen SG contributes a complete reference action independently of g. Concatenation preserves expert identity in separate learned column blocks W_k/R_k. Uniform gating still gives the head all four packets, history-dependent commands, and a full adaptive reference policy. Ordinary concatenation is not a bug, and uniform weights are not an information-destroying average of four actions.

**Constant-gate scale ambiguity:** for any fixed positive c, W'_k=4c_k W_k and R'_k=4c_k R_k yield the same preactivation under uniform g=.25 as the original weights under c. The audit explicitly tested c=(.1,.3,.2,.4) on fixed packets. This proves static parameterization overlap, not that one fixed reparameterization can reproduce arbitrary history-dependent gates or that head compensation caused the observed performance gap.

Static compensation action max error **0**. Complete manual head reconstruction error **1.19e-07**; uniform reconstruction error **1.19e-07**.

**B. Measured local sensitivity, not dead routing.** Perturb raw logits along fixed unit direction(1,−1,.5,−.5), holding history, expert packets, head and a0 fixed.

| Logit displacement | Gate RMS change | Weighted packet RMS | Hidden RMS | Physical action RMS |
|---:|---:|---:|---:|---:|
|0.1|0.009380|0.002396|0.004976|0.002754|
|1.0|0.103703|0.025582|0.055653|0.034808|

No principled sensitivity threshold was prespecified; these modest physical changes cannot by themselves diagnose poor routing. Shared-to-gated-expert hidden norm ratios are measured on8 states per stage, not population estimates:

| t | Shared / gated expert hidden norm |
|---:|---:|
|0|1.00708|
|3|2.11790|
|12|3.34913|
|24|3.73417|
|29|8.22780|

**C. Supported empirical weakness:** saved full-minus-uniform EIG is−.00859149,95%CI[−.0622892,+.0451062],Holm5 p=.75372. Thus useful learned coordination is not demonstrated, not proven harmful or equivalent.

**D. Representation redundancy:** expert packets are not near-identical under the measured CKA and consumed-contribution cosine; those results do not establish or exclude task-specific redundant information. Do not equate gate/head parameterization overlap with four collapsed expert representations.

## Confirmed limitation of existing expert-removal semantics

`disable=k` is applied AFTER all expert outputs feed the coordinator. It zeros g_k and renormalizes the other gates; it does not mask the disabled expert's legacy/response inputs to the router. Consequently the expert can influence final actions indirectly through remaining gates. Controlled test: hold direct packet contributions fixed; zero only disabled expert k's router input blocks; keep g_k=0 in both cases.

| Disabled expert | Disabled weight | Remaining-gate RMS change | Action RMS from its router-input path |
|---|---:|---:|---:|
|E1|0.0|0.028945|0.008120|
|E2|0.0|0.027371|0.011286|
|E3|0.0|0.044496|0.012531|
|E4|0.0|0.013541|0.003371|

This is a **confirmed architectural/ablation limitation** if interpreted as complete expert-pathway removal. It is not automatically a bug: the old intervention is valid as a final-fusion packet knockout. Donor-slot replacement also occurs after gate computation, so it is a fixed-gate packet-content intervention, not a full replacement of expert computation everywhere.

The one-hot reference results are different: routing_override fixes all four weights, cutting the coordinator's influence. Perturbing all unselected first AND response packets leaves the reconstructed action exactly unchanged in every tested one-hot condition. Thus this finding does not invalidate the reference single-pathway or full-vs-uniform scores. Their shared history/SG reference remain intentionally present.

Locations: `B/source/v2_policy.py:28` through `B/source/v2_policy.py:34`, `B/source/v2_policy.py:35`, `B/source/v2_policy.py:37`. No unintended hidden head path outside this explicit graph was detected.
