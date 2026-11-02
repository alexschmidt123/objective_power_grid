# Codex task: implement MoE-sBOED-v5 on UbuntuPC under existing project rules

You are working **on UbuntuPC**, the DEVELOPMENT WORKER for the existing MoE-sBOED project. Your job is to implement **MoE-sBOED-v5** as an explicitly separate, auditable EIG-improvement candidate, building on the actual latest local V4 implementation. Do not silently rewrite V4, alter baselines, promote unvalidated outputs, or claim superiority without evidence. Act autonomously on code inspection, implementation, tests, and bounded diagnostics; do not launch a long formal GPU campaign. Produce a final status including changed files, exact commands, tests, numerical diagnostics, costs, and blockers.

## 0. Non-negotiable project organization and authority

1. On UbuntuPC there must be **exactly two project directories**: stable `objective_power_grid/` and development `objective_power_grid_temp/`. Discover their actual existing paths; do not invent a third project checkout or create another root directory. Development and modifications belong in `objective_power_grid_temp/`; all experimental development outputs there belong in its `scratch_runs/`. Stable `objective_power_grid/experiments/` is the **only** location for permanently accepted experiment results. Promote **validated** source/results from temp to stable following existing promotion scripts/rules; never automatically replace historical data, overwrite results, or promote failing/untested work.
2. **LabPC is authoritative** for formal V1–V4 results and is the sole GitHub publisher. UbuntuPC is a development worker: **NO GitHub push, commit-to-remote, force push, or automatic synchronization**. A local commit is optional only if already standard and approved by repo instructions. Never edit historical LabPC artifacts from here.
3. Respect any existing `AGENTS.md`, `README.md`, repo docs, style/format requirements, configuration schemas, experiment naming, reproducibility manifests, and exact launcher/test conventions **before** editing. Current repo rules outrank guessed structures in this prompt. Use existing timestamp conventions (`MMDDYYYY` experiment layout where applicable). Never create an `experiments/` directory in temp; do not put permanent results in `scratch_runs/`.
4. Official method names are `MoE-sBOED-v1`, `MoE-sBOED-v2`, `MoE-sBOED-v3`, `MoE-sBOED-v4`, and the new `MoE-sBOED-v5`. Treat V5-A (contrastive auxiliary), V5-B (expert credit), and V5-A+B as **ablation/config labels of V5**, not replacements/renames for the canonical previous versions. Do not conflate legacy MoE / ResMoE / SwitchMoE names or older four-expert GitHub code with the latest V4.
5. Do not modify native algorithms, objectives, architectures, training rules, data, or checkpoints of DAD, RL-sBOED, vsOED, ScoreBED, Fixed, Random, or other baselines. You may read their papers/code for concepts only. No copying vsOED actor–critic, Q network, variational posterior, or policy-gradient loop; no baseline teacher/ensemble checkpoint as an expert. No changes in simulator/likelihood/evaluator shared by baselines unless a demonstrated correctness bug is independently tested and separately documented; never silently change the benchmark definition.
6. Do **not** make up missing state, reported seed results, benchmark scores, paths, or old configs. If the authoritative latest **one shared + seven routed, top-3 V4** code and matching checkpoint/config are not available locally, STOP implementation that relies on them, document exactly what is missing, and provide a safe staged integration patch/plan rather than reconstructing V4 from an outdated four-expert implementation.

## 1. First perform a read-only preflight and provenance audit

- Print host, working directories, git branch/status, Python/PyTorch/CUDA/device info, relevant launcher commands, existing tests, and available project instruction files. Check both roots without modifying stable.
- Find the genuine V4 policy class, history/belief encoder, one shared expert, seven routed experts, top-3 selector, router, expert fusion, design head, sPCE loss, explicit likelihood, posterior/contrast sampler, stateful simulation, evaluation and checkpoint serializer. Record actual file paths and SHA256/commit hashes. Check V4 code shape matches its saved checkpoint, not just a description.
- Find the current formal configurations and run manifests for V4 and native vsOED, and identify the **actual benchmark(s)** for the EIG 10.9825/11.0812 results. Do not automatically interpret location finding as IEEE-9/14 or vice versa. Preserve the benchmark-specific observation conditionality, physical-state carry, action bounds, and constraints.
- Reference facts: V4 seed101 **10.9825** nats; seed102 **11.0812** nats; two-seed arithmetic mean **11.03185** nats, not a significance claim. Seed102 improves over paired V1; seed101 paired interval includes zero. UbuntuPC corrected Fixed **8.8972**, old schedule **3.8862** on the same fresh cases; these are distinct versions. Corrected DAD was **not trained** because the acceleration gate failed; do not call an untrained corrected DAD a completed baseline. Retrieve the contemporaneous **actual vsOED scores** from local artifacts and use only matched same-benchmark, same-seed results; if missing, report `vsOED target unresolved` and do not invent a threshold.
- Respect the declared V4 training/evaluation rules: original full-horizon sPCE, **2,000 training contrasts** if confirmed by active config; original **50,000-update native ceiling**, **14,400 charged-seconds LabPC formal run budget** and seed-specific settings if confirmed by manifests; **2,000 held-out trajectories / 1,000,000 evaluation contrasts** and exact paired seeds/bank in the active formal protocol. Do not change these silently or substitute unrelated older one-million-episode protocols. Keep actual resource accounting by episodes, updates, likelihood evaluations, GPU time, wall time, memory and data generation.
- Training seed identities 101–105 remain predeclared where they are part of the active campaign. Development-evaluation seed(s) are separate from reserved final seed **8831001** if the active manifest retains that reservation; never touch reserved final data for model selection. Final checkpoint selection must use the predeclared **final valid budget checkpoint**, not best development EIG.
- Emit a short preflight report to a new timestamped folder inside `objective_power_grid_temp/scratch_runs/` using existing naming; no source edits until this audit confirms the exact V4 implementation.

## 2. Implement independent, behavior-preserving V5 diagnostics FIRST (mandatory gate)

Create versioned diagnostic modules/tests inside the existing repo package layout (do not drop unrelated loose scripts at root). Avoid imports from an unavailable external `/mnt/data` path. Implement/reuse these precise operations:

Given log-likelihood per observation `step_loglik` with shape `[B,T,L+1]`, truth at column 0, one set of **fixed prior negatives** per trajectory across all T stages, and correct conditional log-likelihood `log p(y_t | theta_l, design_t, h_{t-1})`:

```
logL_prefix = step_loglik.cumsum(dim=1)
J_prefix[t] = logL_prefix[t,truth] - logsumexp(logL_prefix[t,:]) + log(L+1)
J_full = J_prefix[T]
r_increment[t] = J_prefix[t] - J_prefix[t-1], J_prefix[0]=0
```

- Assert `sum_t r_increment[t] == J_full` and `J_full == original_V4_sPCE` to numerical tolerance on the same batch, including gradients. **The telescoping decomposition alone gives the exact same total gradient and DOES NOT cure saturation**. Do not market it as an improvement.
- Log full-trajectory saturation: ceiling `log(2001)=7.601402...` nats when L=2000, fraction of scores within 0.01 and 0.10 nats of ceiling, stage at which prefix scores saturate, negative/truth mass distribution, finite gradients, norm of gradients for shared expert, each routed expert, router and design head. Distinguish heuristic truth-likelihood derivative `(1-softmax_true)` from complete backpropagated policy gradient.
- Log overall and stage-stratified expert use/exposure, the three actually selected routed experts, router entropy, top-3 action-space contribution, pairwise functional similarity and action differences, conditional dependence on **realized observations** versus simply experiment stage, ignored/unused experts, effect of shared-expert dominance, and action feasibility.
- Ensure diagnostic hooks do not mutate parameters, optimizer, RNG state used by training (restore/split RNG streams), action sequences, existing V4 loss, or checkpoint. Test same initialized V4 on a controlled synthetic mini-batch before and after instrumentation with identical outputs and gradients and a byte/hash-preservation check. Historical V4 objects remain immutable.
- Use only small **bounded CPU/smoke** diagnostics unless explicitly running an approved development phase. Write a diagnostic result/decision: `saturation_confirmed`, `not_confirmed`, or `inconclusive`, with counts and quantitative thresholds from observations; do not pretend a theoretical bound is empirical proof of saturation.

## 3. V5-A: conditional-posterior-renewed contrastive auxiliary, implemented behind an OFF-by-default gate

Only enable its training in a distinct V5-A configuration **after** genuine V4 saturation is established and posterior sampling passes validation. The main objective must **remain the original full-horizon sPCE**; V5-A adds a clearly labeled auxiliary and thus is a **new training objective**, not mathematically identical training.

For each stage t in a training trajectory, with actual generator truth `theta0`, action `xi_t` and history `h_{t-1}`, obtain negative parameter particles `theta_l^(t)` from `p(theta | h_{t-1})`, conditional on the **same already realized history**. Evaluate their correct next observation likelihood under the same carried physical state/history:

```
C_t = log p(y_t | theta0, xi_t, h_{t-1})
      - logmeanexp_{l=0..L} p(y_t | theta_l^(t), xi_t, h_{t-1})
J_cond = mean_batch sum_t C_t
J_v5a = J_original_full_sPCE + alpha * J_cond
```

- Never draw all stagewise negatives from the prior and falsely call them posterior negatives. Do not reuse a single fixed set of negatives while claiming they were renewed. Posterior resample-move SMC or verified MCMC is acceptable **only if** effective sample size, mixing/correlation, identity with the actual likelihood, proposal/weights, support and label symmetries are checked. If exact independent conditional posterior samples cannot be produced, **label `J_cond` as approximate heuristic**, not as a proven EIG lower bound. If the auxiliary violates caps or validation, keep it disabled and record blocker; NEVER fake successful training.
- The theoretical per-step lower bound requires correct conditional posterior draws and likelihoods. Even with exact draws, **backpropagating through history-conditioned posterior sampling** is nontrivial. Document whether gradients propagate through the sampling distribution; do not claim an unbiased full-policy gradient if sampling/resampling is stopped or discrete. Keep direct full-horizon sPCE gradients as an anchor; test gradient stability and whether the auxiliary actually affects the intended parameters.
- For no-reset dynamical systems, carry the physical state correctly. For conditionally independent observation models, obey their actual likelihood contracts; do not conflate domains. Do not introduce information leakage from the true theta or future observations into the deployed router/experts.
- Implement configuration: `v5.mode=diagnostics|conditional|expert_credit|combined`, `v5.conditional.enabled=false` by default, `alpha`, number of contrasts, posterior sampler method/diagnostics, stage sampling fraction, work cap. For a minimal predeclared design consider `alpha=0.1` versus `alpha=0`, but do not search a wide hyperparameter grid or choose settings from held-out test EIG. Keep schema-compatible YAML and manifest serialization; count *every* extra simulator and likelihood call.
- Tests: oracle Gaussian toy with analytical posterior, conditional score finite and correct dimensions, serial vs vectorized agreement, baseline V4 alpha=0 exact equivalence, per-stage ceiling, no future-data leakage, posterior ESS/quality rejection when insufficient, no-reset history correctness, gradient/cost logging.

## 4. V5-B: genuine MoE-specific expert credit without copying baseline RL/vsOED

Preserve one ALWAYS-ACTIVE shared expert + **seven independent trainable routed experts** + **top-3** expert selection, same encoder, action transform, original sPCE as primary objective. Do NOT replace the mechanism with a critic, learned Q network, teacher policy or a dense ensemble. Keep existing V4 counterfactual/routing improvements unless audit reveals a bug; add only independently testable V5-B mechanisms.

- At sampled **matched histories**, save exactly the same current history, physical state (if applicable), model truth and experiment noise stream. Produce the current top-3 routing choice; construct an alternative by replacing **one** predeclared active routed slot with a predeclared unselected expert. Hold the other two selected slots, all network weights, initial state and random numbers constant. Recompute the mixture output and physical experiment, then follow the **same defined continuation policy** (with history updated under each branch) for the rest of the horizon. Evaluate both branches using a *matched*, adequate-resolution development full-horizon sPCE scoring bank. Counterfactuals are **offline training/development only**; never increase online inference cost.
- Estimate paired full-horizon advantage `A_{i,j,t} = J_T(route with i) - J_T(route with j)` across several independently sampled matched trajectories/noises. Use common random numbers where valid, and distinguish per-history noisy estimates from reproducible advantages. Avoid a biased winner's-curse from maximization over many alternatives without independent re-evaluation.
- Add the resulting router preference/credit as a **clearly specified auxiliary** only on offline training histories (e.g., confidence-weighted pairwise preference) and log its coefficient/normalization. Ensure **top-3 router selection actually receives gradient/update exposure**, including the hard-selection boundary. A soft auxiliary cannot be claimed to optimize excluded experts if they never receive gradients. A controlled exposure mechanism may cover starved experts in training; do not mandate uniform deployment use.
- Encourage **functional**, action-level expert differentiation on histories where experts actually matter, not merely dissimilar hidden embeddings. Any diversity regularizer must be small/ablatable and must not force arbitrary or unsafe action divergence or assign human labels such as "inertia expert"/"damping expert". Do not simply add new experts/parameters to chase EIG; preserve V4 architecture in the primary comparison.
- A specialized expert must demonstrate a **positive reproducible matched-history held-out counterfactual EIG advantage** on a meaningful subset of histories, with different experts advantageous on different history groups, replicated across training seeds. Merely unequal routing or diverse actions is insufficient. Also compare stage-only router vs full observation-conditioned router and learned vs shuffled/constant routing; report starving/collapsed expert counts and matched compute.
- Tests: top-3 exactly, shared always active, no gradient starvation under exposure procedure, deterministic fixed-seed counterfactual reconstruction, replacement of only one slot, valid feasible actions, no reset/state leak, identical policy when credit coefficient zero, no extra online computation.

## 5. Controlled ablations, acceptance gates, and stopping rules

Use V4 unchanged as the reference and register V5 configurations for: `diagnostics-only`, `V5-A only`, `V5-B only`, and `V5-A+B` **only after both single changes individually pass**. Do not silently run combined. Record exact source/config/checkpoint hashes; validate no baseline edits with git diff; formal new method IDs remain unambiguous. Do NOT alter old seed results, rename seed101/102, or select favorable checkpoints.

The required comparisons are:
- Main performance: V5 vs same-benchmark **native vsOED** and V4 using identically paired cases/truth/noise/contrast-bank, same EIG estimator, original evaluation protocol. Report individual seed101–105, EIG (nats), MC SE on held-out trajectories, **paired** differences and CIs; separately report across-training-seed dispersion and uncertainty. Only a complete, independent multi-seed result supports a replicable improvement claim.
- Architecture/control: V4 vs V5; disabling auxiliary and disabling expert credit; random/fixed/router-history-shuffled controls; matched-size single/dense policy only if such ablation belongs in project protocol; expert removal/swaps, stage-vs-observation routing and specialization heatmap. Never modify baseline method implementations for controls.
- EIG and resource accounting: training episodes, main contrast count, extra conditional/posterior contrast likelihoods, state simulator calls, gradient/optimizer updates, held-out scoring work, RAM/VRAM, charged GPU seconds, wall seconds, online inference latency. If the protocol imposes a resource cap, enforce it, do not omit auxiliary work or silently exceed it, and report `INCOMPLETE` if cut short. Run no full GPU campaign during this task.
- Uncertainty and benchmark: use only corrected Fixed (**8.8972**, not old **3.8862**) in any matching summary; original DAD completed at **772 updates**, but corrected DAD *not launched due to failed acceleration gate*—display the blocker, don't report a hypothetical corrected score. Read exact active case identities before comparing.
- Formal objective must remain higher held-out EIG, **not** merely higher training lower bound, higher saturation score, or better diversity. Criteria for a **successful** V5: positive credible paired gain over BOTH V4 and vsOED across independent seeds, documented extra compute, and genuine held-out counterfactual specialization; otherwise report a negative/inconclusive result. No one can guarantee beating vsOED in advance.

## 6. Unit tests, bounded implementation verification, formatting and delivery

Respect current repository-native `pytest`/launcher/config conventions. Use standard method naming and folder layout; do not add extra root project dirs. Add focused tests and smoke scripts to the existing test/tool trees. The minimum correctness gates are:
- V4 original checkpoint loads and emits byte/tolerance-equivalent actions, sPCE and gradients with all V5 switches off.
- Full-horizon sPCE prefix/telescoping equality and correct truth/contrast indices; finite logsumexp numerics; full-horizon ceiling at correct L.
- V5-A posterior negative conditioning, sampler quality rejection, stagewise likelihood correctness and no leakage, alpha=0 equivalence, cost logging, finite gradients.
- V5-B shared + 7 + top-3, feasible bounded action, no accidental forced expert role, credit=0 equivalence, independent expert differentiation hooks, valid counterfactual continuation.
- Simulator dynamics and no-reset state carry (when relevant), all pre-existing tests, checkpoint roundtrip and configuration/manifest provenance.
- Smoke only: <= 16 trajectories, <= 32 negatives, <= 3 updates on CPU/GPU if possible; no long jobs or unapproved timing budgets. These are **correctness checks, not EIG performance results**.

Put development artifacts only in the existing timestamped `objective_power_grid_temp/scratch_runs/...` format. Suggested deliverables (adapt names to mandatory repo format):

```
preflight_and_provenance.md
v4_saturation_audit.json
v5_implementation_notes.md
v5_ablation_configs.yaml (or native separate config files)
unit_test_output.txt
bounded_smoke_output.txt
expert_specialization_audit.csv
compute_accounting.csv
comparison_schema.csv        # schema only until real matched evaluation exists
blockers_and_next_commands.md
```

When tests pass, prepare a **promotion manifest** listing exact validated files and hashes and a non-overwriting path to copy to stable. Do not automatically start formal jobs on UbuntuPC or push GitHub. If current V4 source/code is absent on UbuntuPC, stop after preflight and report safe transfer requirements from LabPC; do not claim V5 implementation.

## 7. Final response format (mandatory)

Use the existing project status-report style, with these exact substantive sections:

1. **Repository & environment** — exact paths, source commit/hash, Git status, active benchmark and available V4 checkpoints.
2. **Implementation** — exact files added/modified, V5 switches/modes, mathematical objective, how V4 was preserved.
3. **Correctness** — exact test commands/counts/results and equivalence checks.
4. **Diagnostics** — measured V4 saturation, gradients, routing/expert usage and valid counterfactual effects; if not measured say `NOT MEASURED`.
5. **Resources** — episode/contrast/posterior/simulator work and wall/GPU time for every smoke/training action.
6. **Performance** — only genuinely run held-out scores; V4 10.9825/11.0812 as previously reported reference, vsOED only when verified, NO fabricated new V5 EIG.
7. **Blockers, promotion status, next reproducible commands** — do not assert formal success before multi-seed evidence.

Proceed now: inspect both local project folders and their project instructions, implement diagnostics first, and implement gated V5-A/V5-B only as feasible under the correctness gates. Do not ask me to reconfirm already supplied constraints; report a blocker rather than inventing missing code/data.
