# Correctness and audit

10 dedicated CPU tests passed in1.28s: gain bounds/count, v1 equivalence at gain1, full-horizon sPCE formula/earliest action gradient, gain finite difference, gate mass and selected expert gradients, feasibility/reproducibility, swap sensitivity, strict checkpoint roundtrip, forward-only deployment and stable clipped updates.

Native RTX4090 smoke: two full B2000/L2000/H30 updates, finite loss, all component gradient groups nonzero, all actions feasible, seven expert gradients observed across both steps, physical swaps affect actions. Weights discarded. readiness/GPU_smoke.json records measured1.1484s training+feature analysis and11.27GB peak allocation; training-only calibration is separately measured.

No gate-normalization bug was found or claimed. No numerical correction to the validated v1 model/objective/optimizer/clipping was introduced. Group preclip/postclip norms and exact effective expert-update counters are observational logging additions.

The old IR report-writer failure is unrelated and its source/checkpoints remain untouched. Frozen source/config hashes guard this new campaign. Confirmation seed96251007 is checked against existing LabPC metadata and reserved before main training. Official seed1001 is a previously consumed historical bank; no claim of untouched testing there.

Limitations: score gradients at top-k rank boundaries are not defined; deterministic v1 semantics are deliberately preserved. Smoke gradient participation does not guarantee later meaningful routed contributions. Conditional posterior ESS is only a reliability screen, not a convergence proof.
