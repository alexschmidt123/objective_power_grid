# Current MoE-sBOED project scope

Official family: **MoE-sBOED**, based on DeepSeekMoE. The sole registered version is **MoE-sBOED-v1**, foundation `13681d7091189a4e68056e6a2864dc58f9d5700b`.

The previous ten-method SIR comparison included historical MoE variants that are not valid methods in the current DeepSeekMoE-sBOED research project. The corrected study contains six baselines and one DeepSeekMoE-sBOED candidate.

| Current method | Location-finding | SIR ODE |
|---|---|---|
| Random | Conventional baseline | Conventional baseline |
| Optimized Fixed | Conventional baseline | Conventional baseline |
| Corrected DAD | Independent baseline | Strict corrected protocol; verify each checkpoint |
| Corrected RL-sBOED | Independent baseline | Strict native protocol; verify each checkpoint |
| Corrected vsOED | Independent baseline | Strict native protocol; verify each checkpoint |
| Corrected ScoreBED | Independent baseline | Corrected potential-head protocol; verify each checkpoint |
| MoE-sBOED-v1 | Structurally validated | PENDING independent adaptation validation and registration |

This is a method-scope table, not a claim that all baseline training/evaluation is complete. A location checkpoint cannot be used for SIR. Conventional baseline method labels never certify an arbitrary historical checkpoint. Preserve each system's corrected implementation, configuration, source/checkpoint hashes, physical model and evaluation-bank identity. The separate strict SIR recovery at `16632ae4b00703ffb1cc4d9b6a956e4a953f00c7` remains a frozen, independently managed baseline-only workload; this cleanup did not edit it or start work there.

## Required MoE computation

An observable history is encoded with the DAD sum encoder and known stage. One always-active shared SwiGLU feature expert and seven independent routed SwiGLU experts feed a shared design head. A learned router selects exactly three routed experts per history. Gate weights are their original **full-softmax probabilities, without renormalizing the selected three**. The feasible action map produces the next experiment directly.

Router and experts are trained offline with the terminal full-horizon sequential sPCE objective and the validated v1 balancing term. DAD history encoding and sPCE are legitimate foundations. A genuine trainable routed feature computation distinguishes this policy from ordinary DAD; sharing its encoder does not invalidate it.

Deployment has no EIG candidate search, action-optimization loop, posterior-EIG maximization, teacher selection, imitation call or neural parameter update. Likelihood, posterior and simulator work during offline learning/evaluation remains legitimate. Router probabilities must be computed before selecting experts; the conceptual pipeline is history → encoder/context → router and shared/routed feature computation → shared head → feasible design.

## Structural validity and scientific evidence

V1 is structurally valid. Useful expert specialization and learned-routing information advantages remain **unproven**. The historical location reference is 10.98906590 ± 0.04479411 nats; this cleanup performed no new official EIG evaluation. Do not infer expert competence from weights, feature differences, activation frequencies or action changes alone. Future candidates require full-horizon EIG, expert interventions, learned/uniform routing, comparable dense controls and training/inference cost evidence.

Versions v2/v3/v4 are unregistered placeholders. FH-v1/v2/v3, RC and IR retain their historical names and are not automatically renamed. The gradient variance audit is a diagnostic, not a registered model. No further gradient-clustering trainer or FH-v4 campaign is authorized.

## Safe active entry points

* `bash run.sh --list-current-methods` lists the seven scoped methods without running an experiment.
* `configs/current_location.yaml` and `configs/current_sir.yaml` list six conventional baselines plus explicit target readiness. They preserve baseline numerical settings; they do not silently dispatch a target into a generic MoE.
* Current v1 training dispatch is `--moe-sboed-v1` with the immutable `configs/deepseek_sboed_location.yaml`, mapped to its original native driver. Its original driver includes scientific controls; controls are not registered benchmark methods. Training requires separate authorization and the local device lock.
* `src.research.current_moe_evaluation` is an explicit frozen-v1 evaluation entry point, with registered digest and benchmark-physics validation. No evaluation was invoked in this task.
* Generic `--methods moe`, residual/switch methods, archived configurations and exploratory launcher flags fail before output allocation. Generic `--methods moe_sboed_v1` directs the caller to the verified native entry point rather than entering old code.

The original native driver, objective, benchmark, encoder, policy and evaluator remain byte-identical to the foundation. Its internal serialization tag `DeepSeekMoE-sBOED` is preserved for checkpoint compatibility; current display/registry name is MoE-sBOED-v1.

## Historical records and machine rules

`historical/project_scope_20261026/manifest.json` verifies byte-identical originals of updated files, including manuscripts and old launchers. `ARCHIVE_INDEX.md` describes the archive; its README is the original README. Existing dated `docs/research_records`, `documents/pimoe*`, `documents/sa_moe`, `reports/FH_*`, checkpoints and experiment directories are historical evidence, excluded by this scope and the exclusion inventory. They retain their original results/names. Archived code is for explicit historical inspection, not a supported current launcher.

Read old checkpoint metadata with `python -m src.research.project_scope inspect-archive PATH`; this reads tensors/metadata without deploying a policy or simulator. Legacy class definitions remain importable for provenance, while active dispatch rejects their use. A registered v1 load pins the original checkpoint SHA256; tensor shapes or a method string alone cannot register a new model.

Develop only in isolated UbuntuPC worktrees. Integrate and commit/push only from an isolated LabPC worktree. Do not edit running source trees, preempt experiments, overwrite checkpoints or merge main automatically. This cleanup authorizes no full training or official EIG evaluation.
