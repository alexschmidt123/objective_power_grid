# MoE-sBOED family: implementation and validation

MoE-sBOED is both the family name and the original per-step full-expert variant.
The four canonical methods are listed below. This implementation adds no new EIG
estimator, inference network, critic, teacher loss, or MSE training objective.

| Component | MoE-sBOED (`moe`) | ResMoE-sBOED (`resmoe`) | SwitchMoE-sBOED (`switchmoe`) | SwitchResMoE-sBOED (`switchresmoe`) |
|---|---|---|---|---|
| Shared pretrained policy base | No | Yes | No | Yes |
| Full-policy experts | 4 | 0 | 4 | 0 |
| Residual experts | 0 | 4 | 0 | 4 |
| DSelect-1 | Yes | Yes | Yes | Yes |
| Learned switching | No | No | Yes | Yes |
| Full-horizon sPCE | Yes | Yes | Yes | Yes |

## Repository audit and changes

| Component | Location | Audited status | Implementation action |
|---|---|---|---|
| Original sparse MoE, DSelect-1, clipping and schedules | `src/research/sparse_moe.py` | Working | Preserved unchanged |
| Corrected location sum encoder | `SparseMoESBOED.context`, same file | Working, 16 features padded plus stage to 193 | Reused unchanged for full experts |
| Historical Strategy-MoE | `src/research/strategy_moe.py` | Working, requested relaxed recurrence | Preserved unchanged; canonical adapter exposes SwitchMoE |
| Switching gate and StrategyManager | Same file | Differentiable previous state, sigmoid gate; no duration constraint | Reused in both switching variants |
| Strong DAD-MLP control | `src/research/moe_architecture_controls.py:ArchitectureControl` | Working sum_single; native head input772 | Entire encoder/head/first experiment restored |
| Matched-input single control | `src/research/matched_sum_single.py` | Implemented; full performance experiment not run | Retained as separate193-input control |
| Residual variants | `src/research/moe_family.py` | Previously missing | Implemented, independently initialized64/64 tanh corrections |
| Shared family factory, names, loading | Same file | Previously missing | Four IDs, aliases, checkpoint-contained base weights, physics checks |
| Common family trainer | `src/research/family_training.py` | Previously missing | One loop for all four, existing rollout/sPCE/clipping reused |
| Root training/evaluation | `src/research/runner.py`, `src/online_cli.py` | Working legacy infrastructure | Add `--variant`, explicit base loading, frozen evaluation; old commands retained |
| Matched additional DAD-MLP training | `train_control`, architecture controls | Previously missing | Optional explicit initial checkpoint; no repeated quarter-budget learning-curve evaluations |
| Persistence and counterfactual diagnostics | `tools/report_moe_family.py` | Prior reports incomplete | Durations, utilization, stage changes, matched proposals, observation sensitivity, gradients |
| Checkpoint compatibility | `tools/validate_family_checkpoints.py` | Historical checkpoints available locally | Read-only paired replay and measured parameter report |
| Tests | `tools/tests/test_moe_family.py` plus existing suites | Two variants existed | Four-variant equivalence, execution, gradients, persistence and roundtrips |
| SIR | `src/research/benchmarks.py`, family dimension-aware encoder | Explicit likelihood exists | All four classes pass CPU ODE rollouts; full experts pass root smoke |
| PK | No implementation found | User explicitly deferred | Excluded |

Historical names `Strategy-MoE`, `strategy_moe_sboed`, and
`strategy_sum_moe_dselect1_v1` resolve to `switchmoe`. Historical class/module/file
names remain importable so old checkpoints and provenance are identifiable.
New artifacts use canonical IDs/names. Expert IDs E1–E4 map to old indices0–3
and historical slots S0–S3 without reordering weights. No checkpoint was renamed,
reinitialized, overwritten, or retrained to establish the canonical name.

## Architecture and action contract

All expert and residual heads have layers input→64→64→action_dim, tanh hidden
activations. Four full experts remain independent. Location full variants retain
the original193-dimensional history representation and separately learned first
design. The DAD-MLP base retains its complete772-dimensional padded representation;
no193/772 substitution occurs.

Residual policies share the native pretrained base sum encoder. Its16 features
are padded to192 and receive a stage coordinate for193-input router/residual
heads; the original base head still receives772 inputs without an injected stage
coordinate. The encoder is frozen with the base initially. Base fine-tuning is
an explicit separate setting. Thus residual training does not silently replace
or independently train a second base representation.

Residual action logits are `base_logits + alpha * weighted_residual`, alpha1 by
default (unvalidated experimental setting). Final residual layers have independent
normal initialization std0.001, including biases. They are small and nonidentical.
The simulator applies the existing feasible transform exactly once. Location
uses its original sigmoid map to [-4,4]². The base first experiment receives no
residual correction. Disabled corrections reproduce the complete base exactly.

Switching uses the existing gate `b=sigmoid(B(context,q_previous))` and recurrence
`q=(1-b)*q_previous+b*p`. Previous state is not detached. Hard inference thresholds
b at0.5; continued rows skip DSelect and only the active expert executes. Experts
always receive updated histories. The first routed decision initializes via
DSelect. Rerouting requests are distinct from actual changes; choosing the same
expert extends its duration. No switching penalty, gate schedule, minimum duration,
or manual expert role has been added.

SIR uses normalized time/count pairs, its own feasible increasing-time mapping,
and dimension-aware sum encoding. It is run on CPU in the family root trainer
because benchmark prior tensors are CPU-resident. The first root SIR attempt
exposed a CPU/CUDA mismatch; its failure artifact is retained, and the CPU rerun
passed. All four SIR class rollouts passed; residual tests use an explicitly marked
shape-only base fixture. No pretrained SIR DAD-MLP checkpoint is registered, so
SIR residual performance training/loading fails explicitly rather than borrowing
an incompatible location base. PK is deferred by the user.

## Historical performance: no full training during framework validation

Location T30, two-source standard-normal prior, noiseSD0.5, bounds[-4,4]²,
training1million primary episodes/L1024, evaluation2000/L1million, seed1001.
Each reported model has one training seed. These are development results, not
untouched publication results or evidence of superiority across training seeds.

| Method | Historical held-out EIG (nats) | Status |
|---|---:|---|
| DAD reference | 9.76665 | Preserved |
| DAD-MLP / sum_single | 10.66260 | Strongest reported control; native input772 |
| MoE-sBOED, sum + slower sharpening | 9.82944 | Preserved |
| SwitchMoE-sBOED, historical Strategy-MoE | 10.23554 | Preserved; below DAD-MLP by0.42706 |
| ResMoE-sBOED | Not evaluated at full budget | Implementation/smoke only |
| SwitchResMoE-sBOED | Not evaluated at full budget | Implementation/smoke only |

Other baseline implementations and their historical results remain unchanged;
this table is the focused architecture comparison, not a new all-method study.

Reanalysis of the historical paired scores gives SwitchMoE minus DAD-MLP
−0.42706 nats, trajectory MC SE0.05745, conditional95% interval
[−0.53967,−0.31445]. Truth/noise identities and the entire contrast-bank SHA256
match. This interval does not represent variability across training seeds.

The SwitchMoE historical run used the **legacy quadratic routing schedule ending
at80%**, whereas the9.82944 MoE used the30%/95% slower schedule. Their+0.40610
difference does not isolate switching. New family defaults use slower sharpening;
loading a historical checkpoint preserves its learned weights/gamma/gate.

Frozen interventions on that same historical SwitchMoE checkpoint, already
completed before this unification, gave: learned hard10.23554, always reroute
10.23554 exactly, always continue4.21898, learned relaxed10.23826. Forced histories
may be out of distribution; their lower EIG does not prove specialization.

## Correctness and compatibility evidence

Focused suite:61 tests passed on LabPC; a clean staged checkout passes60 with
one expected skip for locally preserved historical model binaries. Small
independent numerical fixtures are committed with provenance. The focused tests (family, legacy switching, matched single, controls,
clipping, schedules, sum encoding, SIR physics, location bounds). Four location
variants passed root smoke (8 episodes each, L8); matched additional DAD-MLP
training also passed root smoke. Smoke scores are not performance results.
SwitchMoE additionally passed a root SIR T3 smoke; no full SIR training ran.

Tests cover all canonical IDs/aliases, full-expert and residual forced-reroute
equivalence in relaxed and hard modes, base-disabled equivalence, updated-history
adaptation, permutation invariance, finite sPCE gradients, sparse expert execution,
component clipping, save/reload, state resets and request/change separation.
Fresh-model action tolerances are1e-12; sPCE tolerances1e-11. Legacy switching tests
also check shared gradients. Hard execution counters verify exactly one expert
per case per routed measurement (the base also executes in residual variants).

Read-only replay of32 complete paired historical trajectories: maximum action
error3.91e-14 for MoE and6.31e-14 for SwitchMoE; maximum observation error6.67e-14
and4.23e-13 respectively, below1e-10. With residuals disabled,32 complete
trajectories match DAD-MLP actions, observations, logits and sPCE **bit-for-bit**.
Checkpoint SHA256 values remain unchanged. This preserves historical results;
we did not rerun2k/L1million to call those old EIG values new evaluations.

A fresh real-base T30/L1024 gradient probe (32 trajectories, **zero updates**)
found nonzero router and all residual-expert gradients; base gradients were zero.
ResMoE router norm0.01120, expert norms0.11244/0.12691/0.15411/0.18881.
SwitchResMoE also had gate norm4.81e-7. However,31/32 training scores were within
0.1 of log1025=6.93245; median6.93244788. This is an important saturation warning
for residual training, not evidence that residual training will improve EIG.
Training contrastives remain1024; no estimator or objective was changed.

## Measured parameters

Unique active parameters were counted using forward hooks on a **single** hard
trajectory; directly accessed first-design parameters are counted at stage0.
Counts include frozen base weights when actually executed. Mixing parameters in
DSelect are counted when executed even though k1 makes their softmax derivative
zero. These are physical parameter counts, including legacy padded weights.

| Variant | Total | Pretrained | Newly initialized/trainable, frozen base | Active first routed | Active later routed (reroute) |
|---|---:|---:|---:|---:|---:|
| MoE-sBOED | 72,544 | 0 | 72,544 | 22,424 | 22,424 |
| ResMoE-sBOED | 126,306 | 58,900 | 67,406 | 76,186 | 76,186 |
| SwitchMoE-sBOED | 72,742 | 0 | 72,742 | 22,424 | 22,622 |
| SwitchResMoE-sBOED | 126,504 | 58,900 | 67,604 | 76,186 | 76,384 |

First measurement executes2 learned parameters in all location variants.
Continued switching decisions skip router execution; their active count must be
measured for that state rather than assumed equal to the reroute count above.

## SwitchMoE diagnostic findings

The separately learned first experiment is excluded from all usage statistics.
Existing2000 trajectories give E1/E2/E3/E4 usage0%/3.466%/83.605%/12.929%.
There are1.999 actual changes per episode on average (median2,max2), but100%
of noninitial decisions request rerouting. The long runs therefore arise from
repeated per-step DSelect choices, **not learned hard continuation**.

| Expert | Mean consecutive duration | Median | Maximum | Final20% updates gradient norm>0.001 |
|---|---:|---:|---:|---:|
| E1 | Unused in held-out rollouts | — | — | 14.4% |
| E2 | 1.006 | 1 | 2 | 100% |
| E3 | 24.246 | 25 | 27 | 100% |
| E4 | 3.750 | 3 | 11 | 100% |

E1 had0% such updates in the final5%.73.35% of episodes use one expert for at
least80% of routed measurements; none use just one expert throughout. All
stage-wise frequencies and duration distributions are saved in machine-readable
outputs. Gate gradients exceeded0.001 in98.85% of updates, so the lack of hard
continuation is not a disconnected computation graph.

Matched proposals on64 saved trajectories/1856 histories differ: mean physical
pairwise distances range1.54–3.41 units. Region frequencies and disagreement by
stage are recorded. This shows different proposals, not proven useful roles.
No exploration/refinement/verification labels are assigned.

For1792 fixed partial histories, resampling the latest observation four times
from the existing likelihood while holding designs/stage/previous expert fixed
changed the reroute probability by mean range9.39e-5, max6.27e-4; hard decisions
never flipped. These are **counterfactual routing diagnostics, not rollout EIG**.
The evidence currently supports expert starvation and an almost-always-rerouting
gate, rather than four meaningful persistent observation-adaptive strategies.

## Reproducible commands (prepared, not full-scale launched)

Run at the repository root. The base path is a historical local artifact; model
binaries/results are intentionally not committed. The four commands below each
train1million primary episodes then use the same common2000/L1million evaluation.
They are separate proposed runs, not an instruction to launch all four together.
Use `--smoke` first. New family defaults are sum/64/64/tanh, batch64,
Adam5e-5/(0.8,0.998), decay0.98 every1000, component norm5, existing1e-4 routing
regularizers, gamma1 through30% then cosine to0.001 at95%, L1024, gate init0.99.
No sharpening schedule is applied to the gate.

```bash
BASE=experiments/10142026/10142026_moe_controlled_diagnosis/controls/models/sum_single.pt
bash run.sh --config configs/location_finding_eig.yaml --variant moe --seed 101 --eval-seeds 1001
bash run.sh --config configs/location_finding_eig.yaml --variant resmoe --base-checkpoint "$BASE" --seed 101 --eval-seeds 1001
bash run.sh --config configs/location_finding_eig.yaml --variant switchmoe --strategy-routing-mode learned_switch --seed 101 --eval-seeds 1001
bash run.sh --config configs/location_finding_eig.yaml --variant switchresmoe --base-checkpoint "$BASE" --strategy-routing-mode learned_switch --seed 101 --eval-seeds 1001
```

Optional `--base-finetune` is a separate controlled residual setting. The initial
configuration freezes the base. Matched additional single-policy training:

```bash
bash run.sh --config configs/location_finding_eig.yaml --method sum_single --dad-mlp-continue-from "$BASE" --seed 101 --eval-seeds 1001
```

Residual metadata records the base's1million episodes/729.80seconds plus new
training effort. The continued DAD-MLP control also records base and additional
training separately. A residual gain over the original base alone cannot isolate
architecture from extra training. Report actual wall-clock cost, not only episodes.

Evaluate a completed model without training using the same root/common evaluator:

```bash
bash run.sh --config configs/location_finding_eig.yaml --variant switchmoe --family-evaluate-from PATH_TO_MODEL.pt --strategy-routing-mode learned_switch --eval-seeds 1001
# Same checkpoint, frozen diagnostic interventions:
bash run.sh --config configs/location_finding_eig.yaml --variant switchmoe --family-evaluate-from PATH_TO_MODEL.pt --strategy-routing-mode always_reroute --eval-seeds 1001
bash run.sh --config configs/location_finding_eig.yaml --variant switchmoe --family-evaluate-from PATH_TO_MODEL.pt --strategy-routing-mode always_continue --eval-seeds 1001
# Add --relaxed-routing for the relaxed diagnostic; omit for deployed hard routing.
```

The same command template accepts each canonical `--variant` and family checkpoint.
Historical sparse/Strategy-MoE checkpoints load under `moe`/`switchmoe`. Frozen
comparisons must share truth/noise/contrast-bank identities. New publication
confirmation requires untouched evaluation seeds and multiple training seeds;
trajectory Monte Carlo intervals are not training-seed variability.

```bash
PYTHONPATH=. python tools/validate_family_checkpoints.py --output experiments/10152026/10152026_four_variant_validation
PYTHONPATH=. python tools/report_moe_family.py --run EXISTING_RUN --checkpoint MODEL.pt --rows EXISTING_RUN/eval/METHOD_seed1001.json --training-trace EXISTING_RUN/models/METHOD_diagnostics.jsonl --output experiments/10152026/10152026_four_variant_validation/strategy_diagnostics
python -m pytest -q tools/tests/test_moe_family.py tools/tests/test_strategy_moe.py tools/tests/test_matched_sum_single.py tools/tests/test_moe_architecture_controls.py tools/tests/test_sparse_gradient_stabilization.py tools/tests/test_sparse_schedule.py tools/tests/test_sum_history_correction.py tools/tests/test_sir_offline.py tools/tests/test_location_bounded.py
```

All root evaluations retain online execution timings and machine-readable routing
records. The canonical strategy report contains duration distributions, stage-wise
usage, rerouting versus changes, observable-conditional usage, matched-history
proposal distances/regions, residual magnitude/direction, counterfactual switching,
and update/late-phase gradient exposure. Interpretation of relaxed argmax durations
is explicitly separate from hard decisions. Models save code snapshots/manifests,
benchmark physics, encoder dimensions, optimizer/budget, base provenance, routing,
switch settings, checkpoint hashes, and development-seed status.

## One next controlled experiment

Test **frozen-base ResMoE versus continued DAD-MLP**, initialized from the same
10.66260 checkpoint, each with the same additional primary training budget and
unchangedT30/L1024 objective and common paired held-out evaluator. Hypothesis:
small sparse residual corrections add useful design flexibility beyond additional
single-policy training. Hold switching absent so this comparison isolates the
residual mechanism. Report total pretraining+additional episodes and wall costs,
hard/relaxed EIG, expert exposure, residual directions, score saturation and paired
uncertainty. Because the initial bound is nearly saturated, first inspect its
already-supported gradient/saturation diagnostics during a bounded pilot before
committing to the full budget. Do not change contrast counts or add balancing
losses simultaneously. No performance outcome is assumed. This experiment is
prepared, **not launched**. The four-way full training campaign remains unstarted.

After this implementation/validation report, the user explicitly requested
performance tests of the two remaining residual variants. Their separate
seed101/1001 full-budget runs are authorized for sequential execution, with
unchanged L1024 and common2000/L1million evaluation. Timing probes on6400
additional episodes measured12.82seconds (ResMoE) and13.45seconds (SwitchResMoE),
approximately33–35minutes training each when extrapolated. Allow roughly two
hours for both sequential runs including evaluation/diagnostics. The6400-episode
timing models are not reused as full-run initializations. Full-run results remain
pending; no result is inferred from their smoke/timing scores.

`--training-episodes N` explicitly bounds additional episodes for development
family timing/pilot runs; it is not accepted with smoke/pilot/journal modes.
`tools/evaluate_moe_family_routes.py` evaluates frozen relaxed or forced-routing
views using the unchanged common likelihood evaluator and the completed root
run's entire paired population/contrast bank. It avoids repeating secondary
posterior inference. It records paired intervals, identities, and management
statistics; it introduces no new estimator.
