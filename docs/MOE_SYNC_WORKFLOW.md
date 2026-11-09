# LabPC → GitHub → UbuntuPC

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

## Operation

Run `python3 tools/moe_sync.py --message "Describe the validated change" --files <explicit relative paths>` from MacBook. A clean no-change verification uses `python3 tools/moe_sync.py`. The tool never runs training. On the first sync it creates a new clean Ubuntu worktree from the pushed branch, leaving existing modified checkouts untouched. Subsequent updates use `git pull --ff-only`.
