# Correctness evidence

2026-10-28 CPU regression: 23 passed, 2 skipped in 3.52 seconds (v3 suite plus native-v1 suite). Receipt /tmp/moe_v3_correctness.xml is copied into each pilot provenance directory. Final source modifications require a refreshed receipt before launch.

Verified: conditional mixture log density against independent torch MultivariateNormal; component permutation invariance; four distinct operator paths; exactly two sparse expert executions per row; selected-expert/router gradients; router finite differences; terminal sPCE against replayed full likelihood; terminal finite-difference gradient through complete sequential rollout; early history/action/observation dependency; no auxiliary gradient into generating histories/theta; synthetic bimodal proper-score learning; deterministic checkpoint roundtrip; no true-theta forward signature; no online optimizer; domain dimensions and feasibility; dense parameter match; native-v1 regressions.

Location T30 and SIR T5 full unchanged simulator forward/backward smoke tests pass. IEEE9/IEEE14 adapter dimensions and forward/backward tests pass. Full grid physical simulator tests are BLOCKED/SKIPPED because existing PyCUDA import requires nvcc, absent from the current environment. No simulator was modified. Grid end-to-end numerical compatibility remains unverified; no cross-domain EIG performance is claimed.
