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
