# Correctness evidence

2026-10-28 CPU regression: 23 passed, 2 skipped in 3.52 seconds (v3 suite plus native-v1 suite). Receipt /tmp/moe_v3_correctness.xml is copied into each pilot provenance directory. Final source modifications require a refreshed receipt before launch.

Verified: conditional mixture log density against independent torch MultivariateNormal; component permutation invariance; four distinct operator paths; exactly two sparse expert executions per row; selected-expert/router gradients; router finite differences; terminal sPCE against replayed full likelihood; terminal finite-difference gradient through complete sequential rollout; early history/action/observation dependency; no auxiliary gradient into generating histories/theta; synthetic bimodal proper-score learning; deterministic checkpoint roundtrip; no true-theta forward signature; no online optimizer; domain dimensions and feasibility; dense parameter match; native-v1 regressions.

Location T30 and SIR T5 full unchanged simulator forward/backward smoke tests pass. IEEE9/IEEE14 adapter dimensions and forward/backward tests pass. Full grid physical simulator tests are BLOCKED/SKIPPED because existing PyCUDA import requires nvcc, absent from the current environment. No simulator was modified. Grid end-to-end numerical compatibility remains unverified; no cross-domain EIG performance is claimed.


<!-- observed-progress -->
Observed 2026-10-28T15:05:17Z. State **RUNNING**, active arm **v3**, PID 811813.

| Arm | Updates | Charged seconds | Fresh episodes | Scalar likelihood evaluations | State |
|---|---:|---:|---:|---:|---|
| v3 | 250 | 127.674 | 500000 | 30015000000 | TRAINING |
| v1 | 0 | 1.329 | 0 | 0 | READY |
| dense | 0 | 1.328 | 0 | 0 | READY |
| shared | 0 | 1.328 | 0 | 0 | READY |
| noaux | 0 | 1.328 | 0 | 0 | READY |

This is a timestamped observation, not a completion claim. Current atomic status: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_sboed_v3/location_seed303_20261028T150309Z/status.json`.
Source revision: `92e59f9`, with full dirty dependency tree independently frozen by source manifest `40a31e728ccde64dc05bfe9b77bf06a678b1a6076308863cc09788bb46c3fca8` and verified519-file source archive. Root dispatch and recovered dependencies are not represented by the commit alone. No push or main merge.

No matched development EIG, paired improvement, dense/shared comparison, or useful-specialization finding is available yet. Full training is not yet justified. Grid physical interfaces remain blocked by nvcc.
