# V5 implementation and scope

The original one-shared/seven-routed/top3 DeepSeekPolicy is unchanged (8434total,6898active parameters). Completed V4 source/config/checkpoints were read-staged from LabPC and independently hash/parity checked on Ubuntu. The reference is read-only. Old baselines, V1–V4 policy source, simulator, likelihood and evaluator bytes are unchanged. Only two additive OFF-by-default native dispatch/registry hooks modify pre-existing files; exact backups/diffs are preserved.

## Diagnostics

moe_v5_diagnostics.py captures existing per-stage truth-inclusive likelihood tensors and top3 routes in a cloned model; it consumes no additional simulator/RNG calls for likelihood capture. It computes prefix sPCE, signed telescoping increments, truth/negative mass, gradients and sparse exposure. Exact V4 loss, gradients, original model bytes and training RNG are tested unchanged. Action masks and observation perturbations are descriptive frozen diagnostics, not proofs of specialist competence. Reported gradient norms include the original0.001 balancing regularizer; they are not isolated EIG-only norms. The1-q_truth derivative is explicitly not a complete policy gradient.

## V5-A (OFF)

J_total=J_original_full_sPCE + alpha_cond*J_cond. conditional_scores implements conditional truth-inclusive next-observation contrastive scores with per-stage past-only quality receipts. combined_objective(alpha=0) returns the original tensor without invoking a sampler, changing RNG or altering its graph. Exact Gaussian-posterior oracle tests verify conditioning and dimensions. Approximate posterior gates require ESS, unique coverage, mixing, symmetry and proposal/weight checks and label accepted approximations HEURISTIC, not a proven lower bound. No location posterior sampler has been validated or integrated into policy training; its nonzero coefficient is blocked by the launcher. Historical L2000 saturation is empirically confirmed, but that does not validate posterior sampling or a new gradient. Proposed alpha0.1 is a recorded future ablation only; effective alpha0.0.

## V5-B (OFF)

V4 already performs same-prefix one-slot expert substitutions and terminal-information router ranking. V5 does not rename that mechanism. The new component replicates continuation observation noise at the same joint simulated prefix, truth and contrast bank, computes a mean/SE proxy, and applies a predeclared conservative heuristic weight max(0,abs(mean)-2SE)/(abs(mean)+epsilon). It is not a small-sample significance guarantee or exact posterior-conditional utility.

The candidate router loss blends existing V4 ranking with the replicated-noise weighting: (1-eta)L_V4+eta L_replicated, avoiding two independent additive credits. eta0 executes the original V4 add_credit function exactly, including no new draws; proposed eta0.1 remains disabled. The combined router delta is bounded at5percent of the primary router-gradient norm. Other gradients are unchanged. Bounded alternate-route tests verify gradient exposure of a substituted expert, but automatic expert-exposure training is NOT enabled and no starvation cure is claimed. No new deployment parameters or online search.

The two L32 trained-checkpoint credit smoke cases gave zero differences in all4replicates, consistent with an unresolvable saturated low-contrast score. This is an inconclusive correctness smoke, not a measured high-resolution utility advantage. No further variant search was run. Nonzero V5-B efficacy and specialization are unverified.

## Modes and limitations

Native configs register diagnostics,conditional,expert_credit,combined. Both auxiliary enables and coefficients default to false/zero. Combined is rejected until single-variant scientific gates pass. The root launcher supports diagnostics and credit-smoke only, not formal training. There is no production/formal V5 training loop wired to a validated location posterior; full training requires separate integration, costed protocol and explicit authorization. No V5 has been trained/evaluated or promoted.

Tests:23passed, including5existing bounded compatibility tests. The entire old suite was not run because some tests require64contrasts or80optimizer updates, beyond this task's32contrast/3update limits. Earlier14test and22test iterations are retained. No existing tests or algorithms were weakened.
