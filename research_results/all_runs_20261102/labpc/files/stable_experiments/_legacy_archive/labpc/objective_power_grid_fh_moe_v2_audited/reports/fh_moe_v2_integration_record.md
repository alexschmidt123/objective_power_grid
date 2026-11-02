# Independent integration audit, 2026-10-26

The pre-existing Ubuntu development snapshot was inspected and copied into NEW isolated worktrees on both PCs, based on original DeepSeekMoE-v1 commit 13681d7091189a4e68056e6a2864dc58f9d5700b. Dirty original worktrees were left untouched. The archive SHA256 matched on UbuntuPC and LabPC: c0115fb0572a7e031033df288764f4b07ea8e12d3d57da96d0d927e626fa1e66.

Independent CPU verification on the copied source: UbuntuPC 25 passed in 1.77s (one PyTorch scalar-conversion warning); LabPC 25 passed in 1.72s. Commands used the local mocu_optimized Python and `-m pytest -q tools/tests/test_fh_moe_v2.py tools/tests/test_deepseek_sboed.py`. These reruns are additional CPU preparation costs, excluded from the historical development ledger. SSH, inspection and integration wall time were not comprehensively metered. Saved native GPU smoke/stress and pilot records were audited; they were not rerun during integration.

The scientific pilot gate remains FAILED. No primary training, matched-workload replication, official evaluation or confirmation evaluation is authorized by this candidate's gate. No 18000-update checkpoint exists for v2. All seed101/102 EIG, official paired differences, final deployment latency and matched dense comparisons remain unavailable. The pilot does not establish joint usefulness of routed experts and router.

The planned budget is exclusively 14400 seconds charged optimization on LabPC seed101; Ubuntu seed102 would match the completed primary optimizer count without its own time cutoff. This protocol is documented, not executed.

The archive contains source and small reports/diagnostics only. Historical checkpoints, pilot checkpoint, raw evaluation arrays and trajectories remain on their original PCs and are excluded from Git. The recorded evidence summaries are not a replacement for those originals. No historical record was overwritten.

Scientific decision: do not spend full-run compute on this configuration. Existing-expert-exposure seeding may bootstrap nearly identical gradient references; this is a hypothesis consistent with the failure, not an established sole cause. Before another candidate, predeclare and validate whether stable history-predictable full-horizon gradient structure exists independently of current router populations. Do not tune this candidate's thresholds against its observed development scores.

This branch preserves a correctness-tested negative research result; it is not promotion of a scientifically validated improvement. The final Git commit is the immutable source identifier; verify the same identifier in both isolated worktrees. No merge into main.
