"""Write seven evidence-grounded reports from the completed read-only CPU audit."""
import argparse
import hashlib
import json
import pathlib
import shutil
import numpy as np

ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
B=ROOT/'scratch_runs/moe_hec_v2_icml_20261109'
A=ROOT/'scratch_runs/moe_hec_v2_internal_ablation'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def source(file,needle):
    path=B/'source'/file
    for i,line in enumerate(path.read_text().splitlines(),1):
        if needle in line:return f'B/source/{file}:{i}'
    raise ValueError((file,needle))

def matrix(values,rows=None):
    rows=rows or ['E1','E2','E3','E4']
    s='| | E1 | E2 | E3 | E4 |\n|---|---:|---:|---:|---:|\n'
    for name,row in zip(rows,values):s+='|'+name+'|'+'|'.join('undefined' if v is None else f'{v:.5f}' for v in row)+'|\n'
    return s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=pathlib.Path,required=True)
    parser.add_argument('--verification',default='verification_02')
    parser.add_argument('--publish',type=pathlib.Path,required=True)
    args=parser.parse_args();out=args.audit.expanduser().resolve();proof=out/args.verification
    r=json.loads((proof/'MEASURED_AUDIT.json').read_text());assert r['completion']['status']=='PASS'
    f,g,s,n=r['forward'],r['gradients'],r['specialization'],r['native']
    before=json.loads((proof/'INPUT_SHA256_BEFORE.json').read_text())
    after=json.loads((proof/'INPUT_SHA256_AFTER.json').read_text())
    assert before==after and all(digest(pathlib.Path(p))==h for p,h in before.items())
    # Additional saved result identity checks, without inference or new observations.
    files=[B/'evaluation/candidate_scores.npy',B/'evaluation/EIG_RESULTS.json',
           B/'arms/candidate/RESULT.json',B/'arms/candidate/STARTED.json',
           B/'preparation/RESULT.json',B/'CONFIG_LOCKED.json',
           ROOT/'scratch_runs/moe_hec_sboed_20261108/CONFIG_LOCKED.json',
           ROOT/'scratch_runs/moe_specialization_guided_eig_20261108/CONFIG_LOCKED.json']
    extra={str(p):digest(p) for p in files}
    old=json.loads((B/'evaluation/EIG_RESULTS.json').read_text())
    assert digest(B/'evaluation/development_validation_bank.npz')==old['bank_sha256']
    assert np.array_equal(np.load(A/'full_scores.npy'),np.load(B/'evaluation/candidate_scores.npy'))
    cfg=json.loads((B/'CONFIG_LOCKED.json').read_text());train=json.loads((B/'arms/candidate/RESULT.json').read_text())
    prep=json.loads((B/'preparation/RESULT.json').read_text())
    reports={}
    header=f'''Audit date: 2026-11-12. Exact object: frozen V2Actor candidate, native iteration16001, actor SHA256 `{r['actor_identity']['actor_sha']}`. This is not V2.1, RR-MoE, or the later routing pilot.

Source aliases on LabPC: `B={B}`; `A={A}`; `V={r['source_vendor_path']}`. Locations below use these exact aliases and line numbers. Measured CPU evidence: `{proof}/MEASURED_AUDIT.json`. Five deterministic unit tests plus actual-checkpoint tests; {r['completion']['protected_file_count']} protected files hash-preserved, zero optimizer steps, zero new observation simulations, zero GPU jobs. No native objective or historical model/source was changed.

'''
    text='# Architecture and tensor shapes\n\n'+header
    text+='''## Actual computation

```mermaid
flowchart TD
 H["Observed prefix x: Bx117"] --> S["History:117-128-128"]
 H --> Q["Frozen belief:8 GMM components,64 nodes"]
 H --> SG["Frozen SG reference a0:Bx2"]
 S --> C["Commander context144;20 commands"]
 Q --> C
 SG --> C
 C --> E["Four heterogeneous expert packets:Bx4x128"]
 Q --> E
 SG --> E
 H --> O["Frozen critic:three commanded probes"]
 O --> E
 E --> R["Legacy384 coordinator + response256 projection"]
 S --> R
 R --> G["Four soft weights"]
 E --> F["Gate first64 AND response64 per expert"]
 G --> F
 S --> F
 F --> D["Shared128 ReLU;128-2;tanh delta"]
 SG --> ACT["a=a0+4delta-a0 abs(delta)"]
 D --> ACT
 ACT --> OBS["Experiment observation;append prefix"]
 OBS --> H
```

Every expert executes. This is dense soft-gated feature collaboration, not Top1/Top2 execution. The commander changes actual queries; the coordinator changes consumed feature weights. The expert packets are not four complete experimental designs.

## Exact shapes and dtypes

`B` below is batch size, distinct from source alias B. All floating policy tensors/weights are torch.float32. Integer stage, searchsorted and gather indices are torch.int64; masks are bool. Existing evaluation banks/scorer accumulations are float64. Float64 is used only for the audit's numerical derivative check, not to modify model precision.

| Quantity | Shape / transformation |
|---|---|
| Raw history | [B,117]=onehot stage30 + flattened previous designs58 + previous scalar observations29; future slots zero |
| History encoder | [B,117]→[B,128]→[B,128], ReLU after each layer |
| Belief input at decision t>0 | [B,3t] from t past design-observation pairs; frozen net index t−1 |
| Belief outputs | weights[B,8], means/stds[B,8,4];8 diagonal Gaussian components |
| Sigma points / weights | [B,64,4] / [B,64]; deterministic quadrature, not64 independent samples |
| SG reference | [B,117]→[B,2] |
| Commander context | history128 + reference2 + belief summary14=[B,144] |
| Commander | [B,144]→64 Tanh→20; plus stage offset[30,20], then Tanh; commands[B,20] split4/6/6/4 |
| E1 region tokens | [B,8,9]→[B,8,64]→attention sum[B,64] |
| E1 full packet | anchor64 concatenated with live-minus-anchor64=[B,128] |
| E2 exact query |3 actions[B,3,2],64-node signal[B,3,64], token[B,64,4] |
| E2 legacy and response | token→[B,64,64]→pool[B,64]; semantic[B,12]+adapter[B,36]→52 gives response[B,64]; packet[B,128] |
| E3 sensitivity | Fisher[B,4,4]; upper10+direction4=[B,14]→64; anchor64+live-minus-anchor64 packet[B,128] |
| E4 legacy | frozen critic[B,3,119]→[B,3,1]; legacy input[B,14]→64 |
| E4 response | semantic[B,12]+adapter[B,20]→52 gives response[B,64]; packet[B,128] |
| Four packets / diagnostic features | slots[B,4,128]; `features` is first+response[B,4,64], not the complete packet |
| Coordinator context | history128 +4 legacy64=[B,384] |
| Response context |4 response64=[B,256] |
| Raw router logits / gates | [B,4] / [B,4] |
| First fusion | [B,384]→[B,128] from history and gated first channels |
| Response fusion | [B,256]→[B,128], no bias |
| Combined preactivation/head | add128 vectors→ReLU→[B,2]→Tanh delta |
| Final physical design | [B,2], each coordinate bounded[−4,4] |
| Live native critic | [B,119]=history117+action2→256→256→256→1 |
| Live/frozen posterior | after n observations input[B,3n]; trunk256,256; branches256,256→8/32/32; density input theta[B,4]→log q[B] |

At t=0/15/29, x remains[B,117]; there are0/15/29 observed pairs; belief uses analytic prior/[B,45]/[B,87]. After the final observation, terminal posterior net29 consumes[B,90]. Thus net29 is not used to select a31st experiment. E4's legacy remaining=(30−t)/30 includes the current experiment, while prospective future opportunity uses(29−t)/29 and is zero at t29; these are different intended quantities, not evidence of an off-by-one bug.

'''
    text+=f"Actual hook sample:40 saved states (8 trajectories×t0/3/12/24/29). Complete action reconstruction max error **{f['manual_action_max_error']:.3g}**. Sources: `{source('hec_policy.py','class Belief')}`, `{source('hec_policy.py','def context')}`, `{source('v2_policy.py','def compute')}`, `{source('v2_policy.py','def forward')}`. The full hook shape inventory is saved in measured evidence.\n"
    reports['ARCHITECTURE_AND_TENSOR_SHAPES.md']=text

    text='# Expert parameters, provenance and training\n\n'+header
    text+='''## Exact expert implementations

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
'''
    for name in ['region','discriminate','sensitivity','sequential','adapter2','adapter4']:
        z=g['groups'][name];text+=f"|{name}|{z['parameter_elements']}|{z['gradient_norm']:.7g}|{z['initial_final_L2_change']:.7g}|{z['optimizer_step_min']:.0f}–{z['optimizer_step_max']:.0f}|\n"
    text+='\nEvery listed parameter participates in the actor optimizer and has finite gradients. Core experts have3,800 saved native Adam steps:1,800 HEC-v1 phase-C steps plus2,000 V2 steps. New adapters have2,000 native steps; their preparation optimizer is separate. Phase-B audit confirms zero history gradient and nonzero expert gradients.\n\n'
    text+=f"Sources: `{source('hec_policy.py','self.commander=')}`, `{source('v2_policy.py','self.adapter2=')}`, `{source('migrate_v2.py','def phase')}`, `{source('prepare_v2.py','def prepare')}`, `{source('v2_tasks.py','def labels')}`, `{source('train_v2.py','def run_arm')}`. Exact parent hashes are in B/CONFIG_LOCKED.json and the protected manifests.\n"
    reports['EXPERT_PARAMETERS_AND_TRAINING.md']=text

    text='# Router and commander gradient audit\n\n'+header
    text+='''## Exact router and command code

Commander context=[live history128,SG provisional design/4 two,belief mean4,log1p variance4,component-mean correlation6]=144. Commander144→64 Tanh→20; add stage_logits[t]−reference_logits[t], then Tanh. Split4/6/6/4 drives actual region attention, likelihood probes/hypothesis tilts, Fisher location/direction and prospective-information queries. Commands are not routing weights.

Coordinator input=[history128,legacy expert features4×64]=384. `coordinator`:384→64 Tanh→4 (24,900 parameters). `response_gate`:256→4,no bias (1,024). `raw=coordinator(context)+response_gate(flatten(responses))`; `g=softmax(10*tanh(raw/10),dim=-1)`. This produces four positive continuous weights summing to1, not expert execution probabilities. Stage_logits has600 parameters; commander10,580. Both expert channels receive the same g.

Initialization: coordinator inherited trained HEC-v1 weights; it originally had a zero output layer at HEC-v1 creation. New V2 response gate is zero initialized. The existing actor Adam group is expanded/replaced with model.parameters while retaining old compatible objects/moments; new parameters start native Adam fresh. No optimizer omission was detected.

| Module | Native-loss gradient L2 | Weight change L2 | Saved Adam steps | All parameters in optimizer |
|---|---:|---:|---:|---|
'''
    for name in ['commander','coordinator','response_gate']:
        z=g['groups'][name];text+=f"|{name}|{z['gradient_norm']:.8g}|{z['initial_final_L2_change']:.8g}|{z['optimizer_step_min']:.0f}–{z['optimizer_step_max']:.0f}|{z['all_parameters_in_optimizer']}|\n"
    text+=f"\nAll tested gradients are finite and nonzero. Multi-chunk checkpoint recomputation (chunk7 vs direct path) relative gradient error **{g['checkpoint_relative_gradient_error']:.3g}**. This tests the real checkpoint closure and chunk binding, not a toy router. Float64 raw-logit→action finite-difference max error **{f['finite_difference_max_error']:.3g}**; cached float32 expert features and actual head weights were promoted only for this numerical check.\n\n"
    text+='## Variation and sensitivity\n\n| Expert | Mean gate over512×30 histories | Mean within-stage SD |\n|---|---:|---:|\n'
    for k in range(4):text+=f"|E{k+1}|{s['global_gate_mean'][k]:.6f}|{s['mean_within_stage_gate_SD'][k]:.6f}|\n"
    text+='\nVariation is measured, but not used as proof of good routing.\n\n| t (zero based) | Raw squash derivative mean | Gate-to-action Jacobian RMS | Learned-vs-uniform action RMS |\n|---:|---:|---:|---:|\n'
    for t,z in f['stage_metrics'].items():text+=f"|{t}|{z['raw_squash_derivative']:.6f}|{z['gate_to_action_Jacobian_RMS']:.6f}|{z['learned_uniform_action_RMS']:.6f}|\n"
    text+='\nThe tanh-logit transform is not saturated on these40 states, router gradients do not vanish, and changing gates changes actions. These are local sensitivity checks, not evidence that those changes improve EIG. The actual actor objective is−mean live critic Q, not direct differentiation of independently measured full-trajectory EIG; critic ranking/calibration as a cause of routing weakness remains unverified.\n\n'
    text+=f"Locations: `{source('v2_policy.py','commands=')}`, `{source('v2_policy.py','coordctx=')}`, `{source('v2_policy.py','raw=self.ordinary')}`, `{source('migrate_v2.py','a.actor_optimizer.param_groups')}`, `V/pg_vsoed.py:697–720` (native actor loss and update). No native update is executed in this audit.\n"
    reports['ROUTER_GRADIENT_AUDIT.md']=text

    text='# Shared-head bypass, sensitivity and removal semantics\n\n'+header
    text+='''## Exact graph decomposition

Let first_k,response_k each be64-vectors. The128-dimensional head preactivation is

`pre = b + W_history*s + sum_k g_k*(W_k*first_k + R_k*response_k)`.

Then delta=tanh(W_out*ReLU(pre)+b_out) and action=a0+4delta−a0*abs(delta). Define U_k=W_k*first_k+R_k*response_k. Every U_k occupies the same consumed hidden basis; raw expert packet coordinates do not.

**A. Confirmed structural alternative paths, not a coding error.** History enters the fusion directly, and frozen SG contributes a complete reference action independently of g. Concatenation preserves expert identity in separate learned column blocks W_k/R_k. Uniform gating still gives the head all four packets, history-dependent commands, and a full adaptive reference policy. Ordinary concatenation is not a bug, and uniform weights are not an information-destroying average of four actions.

**Constant-gate scale ambiguity:** for any fixed positive c, W'_k=4c_k W_k and R'_k=4c_k R_k yield the same preactivation under uniform g=.25 as the original weights under c. The audit explicitly tested c=(.1,.3,.2,.4) on fixed packets. This proves static parameterization overlap, not that one fixed reparameterization can reproduce arbitrary history-dependent gates or that head compensation caused the observed performance gap.

'''
    text+=f"Static compensation action max error **{f['static_gate_reparameterization_error']:.3g}**. Complete manual head reconstruction error **{f['manual_action_max_error']:.3g}**; uniform reconstruction error **{f['uniform_action_max_error']:.3g}**.\n\n"
    text+='**B. Measured local sensitivity, not dead routing.** Perturb raw logits along fixed unit direction(1,−1,.5,−.5), holding history, expert packets, head and a0 fixed.\n\n| Logit displacement | Gate RMS change | Weighted packet RMS | Hidden RMS | Physical action RMS |\n|---:|---:|---:|---:|---:|\n'
    for amount,z in f['logit_perturbations'].items():text+=f"|{amount}|{z['gate_RMS']:.6f}|{z['weighted_packet_RMS']:.6f}|{z['hidden_RMS']:.6f}|{z['action_RMS']:.6f}|\n"
    text+='\nNo principled sensitivity threshold was prespecified; these modest physical changes cannot by themselves diagnose poor routing. Shared-to-gated-expert hidden norm ratios are measured on8 states per stage, not population estimates:\n\n| t | Shared / gated expert hidden norm |\n|---:|---:|\n'
    for t,z in f['stage_metrics'].items():text+=f"|{t}|{z['shared_to_expert_norm_ratio']:.5f}|\n"
    text+='''
**C. Supported empirical weakness:** saved full-minus-uniform EIG is−.00859149,95%CI[−.0622892,+.0451062],Holm5 p=.75372. Thus useful learned coordination is not demonstrated, not proven harmful or equivalent.

**D. Representation redundancy:** expert packets are not near-identical under the measured CKA and consumed-contribution cosine; those results do not establish or exclude task-specific redundant information. Do not equate gate/head parameterization overlap with four collapsed expert representations.

## Confirmed limitation of existing expert-removal semantics

`disable=k` is applied AFTER all expert outputs feed the coordinator. It zeros g_k and renormalizes the other gates; it does not mask the disabled expert's legacy/response inputs to the router. Consequently the expert can influence final actions indirectly through remaining gates. Controlled test: hold direct packet contributions fixed; zero only disabled expert k's router input blocks; keep g_k=0 in both cases.

| Disabled expert | Disabled weight | Remaining-gate RMS change | Action RMS from its router-input path |
|---|---:|---:|---:|
'''
    for z in f['packet_removal_indirect_router_path']:text+=f"|E{z['expert']}|{z['disabled_weight_max']:.1f}|{z['remaining_gate_RMS']:.6f}|{z['action_RMS_from_disabled_expert_router_input']:.6f}|\n"
    text+='''
This is a **confirmed architectural/ablation limitation** if interpreted as complete expert-pathway removal. It is not automatically a bug: the old intervention is valid as a final-fusion packet knockout. Donor-slot replacement also occurs after gate computation, so it is a fixed-gate packet-content intervention, not a full replacement of expert computation everywhere.

The one-hot reference results are different: routing_override fixes all four weights, cutting the coordinator's influence. Perturbing all unselected first AND response packets leaves the reconstructed action exactly unchanged in every tested one-hot condition. Thus this finding does not invalidate the reference single-pathway or full-vs-uniform scores. Their shared history/SG reference remain intentionally present.

'''
    text+=f"Locations: `{source('v2_policy.py','coordctx=')}` through `{source('v2_policy.py','slots=torch.cat')}`, `{source('v2_policy.py','if replace_expert')}`, `{source('v2_policy.py','first=slots')}`. No unintended hidden head path outside this explicit graph was detected.\n"
    reports['SHARED_HEAD_BYPASS_TEST.md']=text

    text='# Specialization diagnostics\n\n'+header
    text+='''## Distinct computation is verified; useful four-way specialization is not

A+.05 perturbation within each command block changes its own expert packet, with exactly zero packet change in the other three pathways on the tested histories. Expert MLPs consume distinct model-derived inputs, have different widths, have finite gradients, and changed during historical optimization. These facts reject the simple explanation “four untouched copies computing the same thing.” They do not prove four complementary decision strategies.

## Same-history feature similarity

Linear centered kernel alignment(CKA), on512 trajectories×stages3,12,24=1,536 saved states. CKA is invariant to orthogonal feature-basis changes; direct coordinate cosine between independently learned packet spaces would be misleading. Global CKA includes stage variation; per-stage matrices are also saved in measured evidence. Zero-variance features yield undefined rather than a false similarity score.

Full128-dimensional packet CKA:

'''+matrix(s['packet_linear_CKA'])
    text+='\nLearned116-coordinate probe representation CKA (first64+response tail52):\n\n'+matrix(s['learned116_linear_CKA'])
    text+='''
The12 analytic coordinates are excluded for E2/E4 in the validated competency probe. The equal116-dimensional convention also drops12 learned response coordinates for E1/E3; this is a probe-design limitation, not a complete assay of everything encoded by those experts. The `features=first+response` diagnostic field is NOT used as a substitute for the true128-dimensional packet here.

Same-basis cosine of actual consumed hidden contributions U_k, averaged on the same1,536 states:

'''+matrix(s['consumed_hidden_contribution_cosine'])
    text+='''
These are not near-identical contributions. Neither low CKA nor distinct consumed directions proves useful conditional information advantage.

## Existing validated task prediction, independently recomputed

Ridge probes calibrated on256 histories×3 stages; held-out256 histories; standardization and rival selection use calibration only. Standardized MSE is appropriate here, not classification accuracy. Rows=expert representation, columns=geometry/discrimination/Fisher/opportunity task; lower is better.

'''+matrix(s['heldout_task_MSE_expert_rows_task_columns'])
    text+='\nDesignated advantage = held-out loss of the calibration-selected strongest rival minus designated expert loss, averaged within each validation history over3 stages. The saved raw loss arrays reproduce the reported contrasts to1e-12.\n\n| Expert | Designated advantage | 95% CI | Holm4 p |\n|---|---:|---|---:|\n'
    for name,z in s['recomputed_designated_advantages'].items():text+=f"|{name}|{z['mean']:+.6f}|[{z['CI95'][0]:+.6f},{z['CI95'][1]:+.6f}]|{z['Holm4_p']:.6g}|\n"
    text+='\n**E2 and E3 pass designated-task advantages; E1 and E4 do not.** Lack of significance is not proof that E1/E4 contain no useful information. Frozen preparation readouts also fail post-training constant-predictor comparisons:\n\n| Frozen readout | Constant MSE minus readout MSE | Holm2 p |\n|---|---:|---:|\n'
    for name,z in s['existing_competency']['new_readouts_vs_constant'].items():text+=f"|{name}|{z['mean']:+.6f}|{z['Holm2_p']:.6g}|\n"
    text+='''
Negative differences mean the inherited frozen readout is worse than the calibration constant predictor. This supports a lack of retained readout alignment after native training; it does not prove representation forgetting, because a freshly calibrated probe can recover E2's task information. Experts continue changing while readouts are frozen and auxiliary loss=0, by design. The actual amount caused by this schedule is unestablished without a controlled comparison.

## Gradient similarity in a comparable space

Comparing raw parameter gradients across different-sized expert networks is undefined. Instead, isolate each expert's contribution to the actual native critic actor loss and differentiate with respect to the SAME128-dimensional history embedding; detach gates, direct history path and the other expert contributions. Concatenate40 case gradients and take cosine:

'''+matrix(g['isolated_expert_gradient_history_cosine'])
    text+='''
The local gradient directions differ and are finite; this is evidence against identical local computations, not conditional EIG specialization.

## Existing conditional contributions and regime interactions

Recomputed24 saved conditional tests:128 joint-generated prefixes,4 future-noise replicates,stages3/12/24,calibration-median uncertainty regimes. Preserve true latent paired with its observed prefix, change one donor expert packet at the next decision, resimulate future observations under changed actions, then continue V2 normally. Replicates are clustered by history. The score is remaining normalized native variational return, using the restored frozen HEC-v1 posterior at14001; it is NOT independently contrast-scored conditional EIG or exact conditional MI. Gates are computed before donor replacement.

'''
    text+=f"Holm24 significant individual contribution tests: **{s['conditional_significant_count']}/24**. All raw effect means reproduce saved JSON to1e-12. An explicitly exploratory high-minus-low regime interaction analysis on the existing per-case mean effects also finds **{sum(z['Holm12_exploratory_p']<.05 for z in s['exploratory_regime_interactions'].values())}/12** Holm12-significant interactions. This additional family is an audit diagnostic, not a predeclared new EIG result.\n\n"
    text+='''The full model beats each frozen one-hot pathway in the2,000-case EIG study, supporting collaboration within this checkpoint. It does not establish each expert's unique conditional advantage, individual standalone quality, or expert chronology. One-hot/donor interventions can shift feature scales/distributions.

'''
    text+=f"Locations: `{source('evaluate_v2.py','Z=np.concatenate')}`, `{source('evaluate_v2.py','rival=min')}`, `{source('evaluate_v2.py','for j,k in enumerate')}`, `{source('conditional_diagnostics.py','def run_conditional')}`, `{source('conditional_diagnostics.py','score=a.post_approx')}`. New CKA/gradient diagnostics generate no new observations or training data.\n"
    reports['SPECIALIZATION_DIAGNOSTICS.md']=text

    text='# Native sBOED correctness\n\n'+header
    text+='''## Objective and update implementation

The native training objective is full-horizon VARIATIONAL information gain, not exact EIG or training-time PCE. With gamma1, incremental r_t=log q_{t+1}(theta|observed prefix_{t+1})−log q_t(theta|prefix_t), prior at t0; the sum telescopes to terminal log q30−log prior. Actor loss is−mean Q_live(x,actor(x)) over10,000 replay trajectories×30 stages=300,000 state-action rows. Critic MSE target is r_t+Q_target(next history,actor_target(next history)); transition0 gives full TD factor1; terminal critic target sums final reward slots. No next action is used after the30th observation.

Per native iteration:1,000 new T30 episodes,10,000 replay trajectories from1M capacity,5 critic steps,5 steps per each of30 posterior networks,1 actor step; target Polyak tau.1,LR decay.9999,exploration decay.9999. Candidate completed2,000 additional actor updates,10,000 critic updates,300,000 posterior optimizer steps,2M new episodes/60M observations. V2 phase B200 freezes history only; C1800 trains history too. Live native posterior/critic weights update independently; frozen actor providers and anchors do not. All network/provenance distinctions are explicit in the expert report.

Actor parameters remain differentiable through continuous commanded model queries, expert MLPs, router and fusion. Belief extraction and frozen SG action are intentionally under no_grad. Frozen opportunity-critic weights do not train, but its input derivative can flow to commanded actions. Prospective searchsorted node indices have no continuous selection gradient. These are explicit implementation choices, not unnoticed expert/router detachments.

## Chronology and leakage tests

Native form_actor_input and prefix_inputs match exactly at t0/3/12/24/29. Replacing every unseen design and observation by arbitrary large values leaves the packed prefix AND action bitwise unchanged. Actor.forward accepts x only: true simulator theta is not an input. Expert preparation samples approximate-belief latent variables; native posterior/reward fitting legitimately uses simulator theta in training. This is separate from policy-time information. These tests cover the implemented input path, not every possible scientific dataset leakage scenario.

## Physical model and numerical checks

Natural log signal m(theta,a)=log(.1+sum_{j=1}²1/(||theta_j−a||²+1e−4)); y~Normal(m,.5²). theta has4 coordinates, design2,source prior iid standard normal; design bounds[−4,4]². Fisher uses noise multiplier4, consistent with variance.25. Physical output a=a0+4delta−a0 abs(delta),delta∈[−1,1],a0∈[−4,4], preserves bounds. Source deterministic function, expert query function, analytic Jacobian and scorer were compared on saved values.

| Check | Maximum absolute discrepancy |
|---|---:|
'''
    for title,key in [('Native signal vs expert signal','mean_signal_max_error'),('Native vs literal Gaussian log likelihood','native_loglikelihood_max_error'),('Analytic vs autograd parameter Jacobian','analytic_jacobian_max_error'),('Sum rewards vs terminal log q−log prior','variational_reward_telescoping_max_error'),('Frozen scorer vs literal positive+11-negative complete trajectory','literal_scorer_max_error')]:text+=f"|{title}|{n[key]:.12g}|\n"
    text+='''
The native likelihood's tiny normalization discrepancy reflects float32 sigma/normalization constants in a float64 comparison; it cannot explain a .0086-nat observed gap and is not identified as a physical-model bug. The audit originally demanded an unnecessarily strict1e−12 there, then used a precision-appropriate1e−8 assertion without changing native code. Reward telescoping is float32; evaluator likelihood accumulation is float64.

## Evaluation and RNG

Saved main bank:theta[2000,4],noise[2000,30],prior contrasts[1M,4]; its SHA matches EIG_RESULTS.json. Frozen internal full scores are bitwise identical to the original candidate score vector. Positive is included once,denominator1,000,001; literal2-trajectory11-negative test evaluates660 negative observation-density terms and matches the frozen scorer exactly. No contrast subsampling is used in original results. The literal audit uses existing observations, not newly simulated trajectories.

Original evaluation creates banks from dedicated np.random.default_rng seeds distinct from preparation/training; common theta/noise/bank pair policies, and changed policies resimulate observations. The diagnostic bank is separate from the2,000-case bank. HEC-v2 is deterministic; routing_uniforms are for historical stochastic-policy comparators, not categorical selection in V2. No new physical random draw or rollout was generated by this audit. Saved evaluations remain development results, not an untouched final test after repeated architecture development. MCSEs omit seed variability,contrast bias and systematic model error.

'''
    text+=f"Saved live critic and each first posterior Adam step count: **{n['critic_optimizer_step']:.0f}** / **{n['posterior_stage0_optimizer_step']:.0f}**, consistent with5 steps per accumulated16,001 native iterations. This audit executed zero optimizer steps. No native sBOED correctness defect was detected in the tested objective, prefix, likelihood, chronology or estimator paths.\n\n"
    text+=f"Locations: `V/pg_vsoed.py:222–274` (packing),`:608–721` (posterior,reward,critic,actor,target updates); `V/post_approx.py:497–540` (NLL fitting); `V/models.py:43–120` (physical model/likelihood); `{source('native_reward.py','def stateless_rewards')}`, `{source('normalized_reward.py','def normalized_gmm_forward')}`, `{source('frozen_evaluator.py','def contrast_scores')}`, `{source('evaluate_v2.py','rng=np.random.default_rng(CFG')}`.\n"
    reports['NATIVE_SBOED_CORRECTNESS.md']=text

    text='# Root-cause assessment and one targeted next change\n\n'+header
    text+='''## Direct answer

No incorrect expert implementation or disconnected/misnormalized router was found in the tested frozen HEC-v2 paths. The positive findings are **structural overlap between routing and the shared decision system**, **unestablished useful history-dependent coordination**, and **only partial empirical specialization**. Expert representations are measurably different; four collapsed copies are not supported. These findings do not establish the unique causal explanation of the small learned-uniform EIG difference.

Reference scores reproduced from saved arrays:learned12.92935261,uniform12.93794410,best one-hot pathway E4=12.42782313. Learned−uniform−.00859149,95%CI[−.06228921,+.04510623],Holm5 p=.75372165. This is failure to demonstrate superiority, not proof that learned routing is worse or useless.

## Classified conclusions with traceable evidence

| Classification | Conclusion | Exact location | Test/measured result | Supported implication |
|---|---|---|---|---|
'''
    text+=f"|CONFIRMED IMPLEMENTATION BUG|None found in tested expert/router/native paths|B/source/v2_policy.py:18–62;V/pg_vsoed.py:608–721|Reconstruction error{f['manual_action_max_error']:.3g};finite active gradients;checkpoint relative gradient error{g['checkpoint_relative_gradient_error']:.3g};packing/likelihood/estimator pass|These tests reject specific disconnection,masking,normalization and chronology failures;they do not prove all code bug-free.|\n"
    text+=f"|CONFIRMED ARCHITECTURAL LIMITATION|Uniform preserves separate expert identity;fixed gate/head scaling is non-identifiable;direct history and SG reference remain|B/source/v2_policy.py:28–37|Constant-gate reparameterization action error{f['static_gate_reparameterization_error']:.3g};separate expert-specific columns and direct terms traced|Router overlaps with other adaptive computations;static gates need not add unique expressive capability. Actual dynamic compensation as cause remains unproved.|\n"
    text+='|CONFIRMED ARCHITECTURAL LIMITATION|Existing `disable` is only a fusion-packet knockout;disabled expert still feeds router|B/source/v2_policy.py:28–37|With g_k=0,zeroing only its router-input blocks changes remaining gates and actions|Removal/donor diagnostics cannot be read as complete removal of an expert pathway. One-hot/uniform reference scores remain valid.|\n'
    text+='|SUPPORTED EMPIRICAL WEAKNESS|Learned coordination has no established EIG advantage;local action response is modest but nonzero|B/source/v2_policy.py:30–37;A/PAIRED_COMPARISONS.md|Learned−uniform−.00859,CI includes0;p=.75372. A unit raw-logit perturbation changes action RMS.03481|Routing is active but its measured decisions do not establish added utility. Not a vanishing-gradient diagnosis.|\n'
    text+='|SUPPORTED EMPIRICAL WEAKNESS|Only E2/E3 pass designated learned-task advantages;no corrected conditional native-return advantage|B/source/evaluate_v2.py:87–105;B/source/conditional_diagnostics.py:22–49|E2 Holm4 p3.15e−5,E3 1.09e−34;E1/E4 nonsignificant;0/24 conditional tests and0/12 exploratory interactions|Evidence does not support four useful conditional specialists. It does not prove E1/E4 useless or all representations redundant.|\n'
    text+='|UNRESOLVED HYPOTHESIS|Critic misranking,true decision-value redundancy,frozen belief accuracy,coadaptation,optimization/sample budget explain weak routing|B/source/train_v2.py;V/pg_vsoed.py:697–711|Native training optimizes−Q;no controlled experiment here isolates these causal mechanisms|Do not claim any is the proven root cause or that an architecture change will guarantee a win.|\n'
    text+='''
All expert cores and router modules changed between initial14001/final16001 checkpoints. Expert-core Adam counts3800 reflect HEC-v1 C1800 plus V2 all2000;coordinator4000;new response gate2000. Tanh squash derivatives≈.987–.999 on tested states argue against saturation. Distinct-packet CKA≈.044–.516 and consumed contribution cosine≈.119–.297 argue against identical representations, but neither proves task-specific complementary decision value. Shared-to-gated-expert hidden norm ratio reaches8.23 at t29 in this small8-case panel; no causal importance is assigned solely from norms.

## ONE targeted next change: make the pathway-removal diagnostic router-aware

Add a **separately named full-pathway-output knockout**, retaining the historical packet-only `disable` intervention unchanged. In new isolated evaluation code, zero expert k's legacy block BEFORE `coordinator`, zero its response block BEFORE `response_gate`, and zero its first/response fusion contribution (then renormalize the remaining gates). The shared history, commander, frozen SG reference and other experts remain explicitly intact. This removes that expert pathway's output from all its consumers; it does not erase the underlying physical information from shared history or other networks, and it does not make execution sparse.

Why this change: the audit directly demonstrates an unremoved information path, so correcting the interpretation/measurement is justified without guessing a training defect. It gives a cleaner answer to “does this expert's output help conditional decisions?” before another routing redesign. Mask-only head ablations and the new all-consumer intervention should have distinct names and receipts. Deterministic regression should verify that perturbing the knocked-out pathway's outputs cannot change remaining gates or actions.

This is a targeted **diagnostic correction**, not a promised performance fix. It leaves native variational EIG,trained V2 and old reports unchanged. No architecture redesign or retraining is recommended from unsupported causal claims, and none was launched in this audit.

## Scope, preservation and reproducibility

Run `python tools/tests/test_hec_v2_code_audit.py` for five deterministic algebra tests, then `python tools/audit_hec_v2_code.py --output NEW_AUDIT_DIRECTORY` under the existing dellma environment; CUDA is hidden inside the audit process and checkpoints load on CPU. Output directories must be new. The requested seven reports are in the parent audit directory. Full numerical proof and114-file before/after manifests are in verification_02. Initial audit-harness attempts are preserved:an oversized legacy NumPy seed was corrected modulo2**32 before model access;an over-strict float64 likelihood normalization assertion was made precision-appropriate. Neither issue was an HEC-v2 bug or a scientific rerun.

The separate prior counterfactual routing pilot did not improve original/uniform V2 and is not reclassified as success here. This audit performs zero new scientific optimizer steps or observation simulations. No model/runtime/checkpoint/dataset/raw scores enter Git;only safe test/report source and technical reports/compact summaries are synchronized. LabPC remains authoritative;Ubuntu only receives the clean fast-forwarded branch.
'''
    reports['ROOT_CAUSE_AND_NEXT_FIX.md']=text
    # Write under the exact user-requested scratch directory, then publish safe reports.
    for name,text in reports.items():
        p=out/name
        if p.exists():raise FileExistsError(p)
        p.write_text(text)
    assert len(reports)==7
    summary={k:r[k] for k in ['actor_identity','completion','forward','gradients','native','recomputed_EIG_comparisons']}
    summary['specialization']={k:s[k] for k in ['packet_linear_CKA','learned116_linear_CKA','consumed_hidden_contribution_cosine',
        'heldout_task_MSE_expert_rows_task_columns','recomputed_designated_advantages','conditional_significant_count',
        'global_gate_mean','mean_within_stage_gate_SD','exploratory_regime_interactions']}
    summary['additional_identity_checks']=dict(bank_hash_matches=True,original_internal_full_scores_bitwise_identical=True,
                                              protected_inputs=extra)
    summary['reports']={name:digest(out/name) for name in reports}
    write=out/'AUDIT_SUMMARY.json';write.write_text(json.dumps(summary,indent=2)+'\n')
    assert all(digest(pathlib.Path(p))==h for p,h in extra.items())
    publish=args.publish.expanduser().resolve()
    if publish.exists():raise FileExistsError('New publish directory required')
    publish.mkdir(parents=True)
    for name in [*reports,'AUDIT_SUMMARY.json']:shutil.copyfile(out/name,publish/name)
    print(json.dumps(dict(status='REPORTS_COMPLETE',scratch=str(out),published=str(publish),reports=list(reports),
                         preserved=True,trained=False),indent=2))


if __name__=='__main__':main()
