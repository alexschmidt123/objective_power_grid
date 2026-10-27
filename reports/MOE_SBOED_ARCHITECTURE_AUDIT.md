# MoE-sBOED-v1 architecture and deployment audit

## Conclusion

**Structurally valid, useful specialization not established.** This is an architecture/compatibility cleanup, not a new EIG result. Original foundation `13681d7091189a4e68056e6a2864dc58f9d5700b`; cleanup starts from gradient-audit source `04b221d2155a59e2afb9d0108c45b22a5702eb8b` in new isolated worktrees. The six foundation files remain byte-identical: `deepseek_sboed.py`, `deepseek_sboed_experiment.py`, `engine.py`, `dad_location_policy.py`, `benchmarks.py`, `journal_validation.py`.

## Executed computation

The location history has 30 padded four-value slots plus known stage. Each observed raw design/observation token has dimension3, encoded3→256(ReLU)→16 then summed. LayerNorm and a known stage/time vector yield context z. With independent bias-free SwiGLU modules E_s,E_1,…,E_7, the policy computes

`p = softmax(W_router z); K = top3(p); u = z + E_s(z) + sum_{e in K} p_e E_e(z); latent = shared_head(u)`.

The selected gate mass is not renormalized. Sparse row dispatch executes only the selected experts for each history. Experts have16→8 gate/up and8→16 down matrices; shared head is LayerNorm16 thenLinear16→2. The sigmoid feasibility map gives actions in[-4,4]^2. Total8434 parameters, activated6898 per history. Expert parameter objects are independent. Tests observe exactly three routed calls and one shared call per single-history forward; unused experts are not executed.

The original terminal sPCE graph differentiates through all30 simulated experiments and adaptive history. Nonzero finite gradients reach encoder, shared expert, selected routed experts and router, with a deterministic finite-difference router check and terminal DAD gradient reference. Softmax derivatives include the full probability normalization; treating top-k membership as locally fixed is the original method. The existing balance term does not replace the terminal objective.

At stage0, zero context and bias-free experts can yield zero expert-weight gradients for that context. This is not disconnected architecture; nonempty histories and full-horizon tests establish actual gradients. Different expert weights or nonzero gradients do not establish complementary information competence.

## Deployment and compatibility checks

CPU-only tests run on UbuntuPC, then independently on LabPC after checksum transfer. Tests cover the exact seven-method registry; all legacy/future-version rejections; native configuration digest; SIR pending failure; same-shape FH metadata rejection; unregistered checkpoint digest; physical-model identity; archive checksums; baseline registry compatibility; legacy plotting exclusion; launchers failing before work; actual original-checkpoint loading and bitwise output equivalence at stages0,1,10,20,29.

The deployment-only test traverses all30 stage inputs under `torch.inference_mode`, patches Adam/SGD/LBFGS steps, autograd, simulator forward/propagation, likelihood, EIG evaluator and SciPy search entry points to fail if called, verifies sparse calls and feasible actions, and compares every parameter before/after exactly. Policy forward imports no predictive planner, teacher or vsOED policy. No hidden true parameter is supplied as input. Runtime patches cover representative forbidden APIs; source inspection and immutable foundation comparison supplement them.

The original checkpoint is loaded without modification using its original `DeepSeekMoE-sBOED` serialization tag. Current loader requires exact registered SHA256, source commit, full variant/horizon,8434 state elements and expert tensor shapes. Physics identity is checked before current evaluation dispatch. Legacy metadata remains readable explicitly without instantiating/evaluating an old policy.

Two old tests are intentionally not executed: the80-step toy supervised-learning test and the small EIG evaluator test. All selected checks are correctness/unit/deployment tests; no full training, GPU work or official EIG evaluation occurs. Record exact test counts and transfer hashes in `MOE_SBOED_CLEANUP_VALIDATION.json` after the independent LabPC checks.

## Updated active paths and historical boundary

`project_scope.py` defines current identities; `run.sh` rejects legacy flags/configurations before dispatch; `runner.py` and `journal_run.py` expose only conventional baseline loops; `training.py` rejects generic MoE training; `journal_checkpoints.py` requires explicit archive access for legacy loading. `current_moe_evaluation.py` loads only registered v1 with physical-model checks. The general `src/context.py`/`layout.py` no longer advertise/load the generic old MoE; `src/experiment.py` blocks old MoE variants/mechanism entry; `src/results/plots.py` excludes legacy MoE labels. Historical report/campaign generation scripts fail when run as current CLIs, while imported lock/baseline helpers remain available. README, current configs, scope/registry docs and manuscript entry pages point to current methods. Original files are checksum archived.

The original immutable native driver retains its declared scientific controls and source semantics; controls are not registered comparison methods. Low-level historical classes remain for explicit provenance, not as current aliases. Original dated records are not deleted or rewritten. Running strict conventional SIR source is outside this new worktree and was not edited or interrupted.

## Limits and next scientific step

SIR current-MoE validity/result remains PENDING. The earlier isolated adapter is not promoted by this scope cleanup. No enhanced version is registered, no EIG superiority claim is made and no specialist trainer is launched. Future useful-specialization claims need controlled full-horizon interventions, learned-versus-uniform routing and compatible dense-model/cost comparisons. Existing gradient audits do not support multistage gradient clustering; frozen expert competence remains a separate scientific question.
