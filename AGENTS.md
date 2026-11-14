## SIR campaign rejected and stopped by user (2026-10-28)

Current SIR seed101 four-hour campaign is EXCLUDED_UNDERTRAINED. User explicitly requested stopping its remaining execution. Do not restart it or launch replacement training without a new instruction. Preserve completed and partial artifacts. The old limited runs are diagnostics, not adequately trained primary comparisons.

## Storage consolidation policy (2026-10-27; supersedes historical launch authorizations)

All objective_power_grid runs were stopped. Do not resume old queues or launch GPU experiments without a new request. Read reports/EXPERIMENT_STORAGE_POLICY.md and reports/EXPERIMENT_PROMOTION_REGISTRY.md in STABLE. Use only STABLE objective_power_grid and development objective_power_grid_temp. Development outputs go to TEMP/scratch_runs/; official results require scientific validation and explicit registration in STABLE/experiments/RESULT_REGISTRY.json. Legacy archive contents are excluded from current tables. Only LabPC commits/pushes; no checkpoints/datasets in Git. Preserve historical bytes and unique source changes.


## Mandatory MoE-sBOED synchronization workflow — 2026-11-09

Applies to current HEC and subsequent MoE-sBOED development. Supersedes older blanket Ubuntu-access prohibitions; historical frozen experimental protocols remain historical records.

- LabPC is the authoritative development/source machine. GitHub is the central version-control and synchronization service. MacBook operates Codex and the SSH orchestration. UbuntuPC is an authorized code-synchronization worker; computation there requires an explicit task assignment.
- At each validated meaningful code checkpoint: save/test on LabPC, inspect Git status and the actual diff, commit explicitly scoped safe source/configuration/documentation changes with a descriptive message, push normally, fetch/pull the corresponding branch on the clean Ubuntu worktree with --ff-only, and verify identical LabPC/remote/Ubuntu HEAD hashes. Report success or the exact failure. No repeated synchronization permission is required.
- Use tools/moe_sync.py from MacBook. It orchestrates the existing SSH aliases separately; LabPC does not need Ubuntu SSH credentials. Do not copy keys or credentials between machines.
- Do not independently edit research source on UbuntuPC. Never overwrite uncommitted work, auto-stash, force-push, force-reset, or change unrelated branches. Existing dirty worktrees and all training processes remain untouched. Use a separate clean worktree when necessary.
- Never commit credentials, private datasets, checkpoints, large outputs, runtime logs, or machine-specific secrets. Compile changed Python source, parse changed JSON and review the staged patch before a commit.
- The dedicated branch is research/hec-moe-sboed-20261109 in alexschmidt123/objective_power_grid. LabPC worktree: TEMP/development/hec_git_sync. Ubuntu worktree: TEMP/development/hec_worker_sync. TEMP denotes each user's Documents/objective_power_grid_temp. Existing STABLE/TEMP working trees and branches are preserved.
- New code belongs in the LabPC Git worktree. Current HEC scratch sources are frozen by PRETRAIN_MANIFEST.json and must not be edited while their experiment runs. Publish only validated snapshots; use a new isolated experiment for scientific changes. Synchronization does not authorize training restarts, compute campaigns, or changes to budgets/evaluation.
- If any validation, push, pull, status or hash check fails, stop the synchronization, preserve both machines' work and report the error. Do not claim synchronized until all hashes match.


## Mandatory MoE-sBOED evaluation comparability — user instruction 2026-11-14

Primary comparisons must use the exact saved HEC-v2 evaluation for V2, DAD, vsOED and other baselines. Never silently substitute banks, seeds, case counts, contrast counts, scorers, physics or checkpoint variants.

- Canonical bank: TEMP/scratch_runs/moe_hec_v2_icml_20261109/evaluation/development_validation_bank.npz;2000 truth/noise cases and1000000 prior contrasts, byte-identical to V2. Routing uniforms come from the same original evaluation directory.
- Use V2's frozen float64 likelihood scorer, chunk16384, positive truth included once, full T=30, sigma0.5, two normally distributed2D sources, and2D design bounds[-4,4]. Preserve each baseline's validated inference implementation and evaluation-mode behavior.
- Reproduce the entire saved V2 learned/uniform, DAD and vsOED score vectors within1e-9 before accepting comparison results. Also reproduce RL-sBOED and ScoreBED when included. Verify bank, scorer, checkpoints and saved-reference hashes; report paired uncertainty and multiplicity correction.
- Include the current learned candidate, original V3, its uniform and all four standalone experts, V2 learned/uniform, DAD, vsOED and the other established fixed baseline checkpoints. Missing baselines are blockers or explicitly incomplete comparisons, never silent exclusions.
- Different contrast counts or banks are supplementary only; never replace this primary benchmark or compare their absolute EIG values to historical scores. Changing the primary protocol requires explicit user instruction. Do not select checkpoints or tune hyperparameters on the reference evaluation.
- Evaluation matching does not imply training-budget matching. State training provenance and budget differences separately. Historical baselines remain frozen references, not newly promoted official results.
- Preserve all historical artifacts. New evaluation outputs go to a new TEMP/scratch_runs directory. No model training is authorized by this rule itself.

The executable guard and rerun entry point are in development/moe_sboed/hec_canonical_evaluation_20261114/. This rule applies to future MoE-sBOED implementation and result reporting.
