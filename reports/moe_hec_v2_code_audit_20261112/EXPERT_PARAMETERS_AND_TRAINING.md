# Expert parameters, provenance and training

Audit date: 2026-11-12. Exact object: frozen V2Actor candidate, native iteration16001, actor SHA256 `90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108`. This is not V2.1, RR-MoE, or the later routing pilot.

Source aliases on LabPC: `B=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109`; `A=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_internal_ablation`; `V=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101/source/src/research/vendor/vsoed`. Locations below use these exact aliases and line numbers. Measured CPU evidence: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_code_audit/verification_02/MEASURED_AUDIT.json`. Five deterministic unit tests plus actual-checkpoint tests; 114 protected files hash-preserved, zero optimizer steps, zero new observation simulations, zero GPU jobs. No native objective or historical model/source was changed.

## Exact expert implementations

The named experts are `torch.nn.Sequential` modules, not separate custom expert classes. `HECActor.__init__` in B/source/hec_policy.py constructs the legacy MLPs; `V2Actor` in B/source/v2_policy.py adds anchors and response adapters. The enclosing actor class is V2Actor(HECActor).

| Expert | Modules and affine layers | Hidden activation | Live policy parameters | Frozen anchor weights |
|---|---|---|---:|---:|
| E1 geometry | region9→64→64 (2 layers); anchor1 same | SiLU |4,800|4,800 |
| E2 discrimination | discriminate4→32→64 (2); adapter2:36→96→52 (2) |SiLU|2,272+8,596=10,868|0 |
| E3 Fisher | sensitivity14→96→64 (2); anchor3 same |SiLU|7,648|7,648 |
| E4 opportunity | sequential14→64→32→64 (3); adapter4:20→96→52 (2) |SiLU|5,152+7,060=12,212|0 |

All have float32 input/output shapes in ARCHITECTURE_AND_TENSOR_SHAPES.md. Frozen anchors are buffers, not trainable Parameters. Diagnostic heads are separate and not deployed to produce actions: original64→14/9/9/3 readouts total2,275; new52→3/6 response readouts total477; these2,752 parameters are frozen during native V2 training. Actor allocation189,822; requires-grad187,070; unused ordinary alternative516 is included in the latter but receives no candidate-forward gradient. Phase-C policy-active live count186,554.

## Distinct model information and learned tasks

- E1: commanded attention over approximate-belief mixture components; tokens[mu/3,log sd,w]. Target14=CDF at tilted mixture mean4,log1p variances4,correlations6.
- E2: commanded spatial probes and node-hypothesis tilting; exact location signal at64 nodes. V2 adds12 analytic plus/minus-hypothesis response coordinates and a learned52-vector. New3-target readout predicts rotated-probe log1p cross-hypothesis Gaussian separation; legacy9-target preparation used means,variances,third moments.
- E3: analytic log-signal Jacobians and noise-scaled Fisher matrix, commanded location and parameter direction. Target9=log1p eigenvalues4,squared weakest eigenvector4,directional Fisher1.
- E4: frozen critic query differences, exact predictive signal moments and commanded probes;12 analytic approximate prospective-opportunity coordinates plus52 learned response coordinates. New6-target task predicts second-step variational gains3 and tail-critic contrasts3. This is approximate-belief/native-bootstrap information, not exact conditional MI.

These are mathematically different query/input computations, not four cloned networks with identical input. However different computations do not prove complementary conditional decision value. All use the same observed history and model, so shared information is expected.

## Initialization and data provenance

Collaborative candidate10001→SG12001→HEC-v1 candidate14001→this V2 candidate16001. HEC-v1 constructed fresh heterogeneous experts using seed202611099101; copied the trained SG history and retained compatible history Adam; SG reference, approximate belief and opportunity critic became frozen snapshots. Its shared preparation used4,096 training+512 validation replay prefixes, random t0..29,300 Adam0.001 steps×256, half zero/half random commands. Predictive targets come from the approximate belief, not recorded hidden replay theta.

V2 inherits those trained experts, history, commander, coordinator and fusion at14001. E1/E3 anchors copy parent terminal networks and remain frozen. New adapters/readouts use seed202611099202. Response gate starts zero; response fusion starts zero except copied E1/E3 fusion blocks, preserving the inherited action. Stage-logit offsets initially cancel reference logits. V2 preparation trains only adapters2/4 and response readouts:4,096 train+512 validation contexts,300 Adam0.001 steps×256; command modes live/stage/zero/random;4 predictive teacher replicas. All old tensors are asserted unchanged. Existing preparation uses108,876 predictive observations, zero new real episodes; held-out E2 loss2.63555→0.28784 and E4 .82090→.39908.

Native V2 B=200 freezes **history only**, not experts. C=1,800 also trains history. All experts/commands/gates/fusion learn the same native actor objective−mean Q; auxiliary_joint_weight=0,diagnostic readouts frozen. Thus no persistent training constraint enforces the preparation's designated semantics or distinct conditional decision regimes. This is a confirmed training choice, not a bug. Live native critic and posterior nets are separate from frozen actor providers.

## Actual updates and finite gradients

The CPU audit backpropagates the actual native actor loss through the saved live critic at fixed saved histories. It performs no optimizer step. Initial/final tensor differences and saved Adam step states provide independent evidence that historical optimization occurred. Gradient magnitudes across different-sized modules are not directly comparable learning rates or scientific utility.

| Module | Parameters | Native-loss gradient L2 | Initial14001→final16001 weight L2 | Saved Adam steps |
|---|---:|---:|---:|---:|
|region|4800|0.005091729|1.879475|3800–3800|
|discriminate|2272|0.0108274|0.6966685|3800–3800|
|sensitivity|7648|0.007328647|2.731298|3800–3800|
|sequential|5152|0.0584996|0.485846|3800–3800|
|adapter2|8596|0.003615162|3.309239|2000–2000|
|adapter4|7060|0.003418826|3.351429|2000–2000|

Every listed parameter participates in the actor optimizer and has finite gradients. Core experts have3,800 saved native Adam steps:1,800 HEC-v1 phase-C steps plus2,000 V2 steps. New adapters have2,000 native steps; their preparation optimizer is separate. Phase-B audit confirms zero history gradient and nonzero expert gradients.

Sources: `B/source/hec_policy.py:44`, `B/source/v2_policy.py:14`, `B/source/migrate_v2.py:5`, `B/source/prepare_v2.py:24`, `B/source/v2_tasks.py:12`, `B/source/train_v2.py:29`. Exact parent hashes are in B/CONFIG_LOCKED.json and the protected manifests.
