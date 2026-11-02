# Continuation of existing Ubuntu preflight

Previous preflight: `/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10312026_052715_moe_sboed_v5_ubuntu_preflight/`; preserved without rerunning/replacing it. The new prompt's suggested undated preflight alias did not exist; the actual timestamped artifact was used.

## Repository and transport

Ubuntu TEMP `/home/qiqi/Documents/objective_power_grid_temp`; STABLE `/home/qiqi/Documents/objective_power_grid`; HEAD6a592c8d2ad10eb890f1d351e579ee4129939227 with pre-existing dirty state preserved. Actual read route: the MacBook's configured BatchMode SSH alias `labpc`, timeout15seconds. The Ubuntu shell was not assumed to have LabPC credentials. All LabPC commands were targeted ls/cat/stat/sha256sum and Git read operations; no remote Python/Torch/CUDA, jobs, writes, permissions, checkout or push. Small completed files were relayed LabPC→MacBook→Ubuntu. The allowlist reference and logs are now read-only on Ubuntu. LabPC active baseline files/logs were not altered or copied.

Completed V4 origin: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_sboed_v4_formal_seeds101_102/`. Current read-only LabPC HEAD763a28dc38401fb9ac63f52bc2751338624c2b98; exact formal source freeze base92e59f99c94287df662e43972a06f89740e8a383 plus its complete dirty-worktree archive/per-file hashes, not an assumption that HEAD equals trained code. Frozen archive SHA256118e34017421fa450866a6e4acf9da54cf3f08444f9643e47902d917eff68b66. No separate textual historical dirty.patch was present; the immutable full snapshot contains exact dirty source bytes. Current unrelated LabPC dirty baseline edits were excluded.

Seed101 checkpoint SHA256d4b8c7fe3f80d650832e556fbbd112b4901b34d01b8fed96ec737db4ff22d5de; seed102d470f54959a0b91e2708d429bf06474777d1c9d85739f572010cdcf69f518860. Checkpoint→configuration→source-manifest links, sizes, and source timestamps are recorded. Pre/post transfer hashes matched for all23core completed files and both supplemental completed logs.

## Parity and active benchmark

V4 CPU replay parity established: independent completed cpu_roundtrip.pt weights and cpu_tests.json utility/router-gradient receipt reproduced; both trained checkpoints strictly load and match frozen source rollout actions, full-horizon scores and gradients on Ubuntu. Maximum recorded trained-checkpoint gradient discrepancy0. This establishes the recorded CPU parity scope; it does not rerun the full1M-contrast evaluation or prove cross-GPU numerical identity.

The verified benchmark is two-source location-finding T30, independent standard-normal coordinates, log-signal Gaussian sigma0.5, bounds[-4,4]^2. Physical state is empty/conditionally independent here. No SIR/grid EIG is used. Active formal settings are top-level B2000,L2000,max50000updates,14400charged seconds,final valid checkpoint; legacy1M-episode wording nested in the benchmark metadata is stale and is not a training quota. Original final evaluation is2000cases/L1M/seed1001. No original or reserved8831001 evaluation draws were generated here.

The native vsOED seed101 saved development result is 12.48084689±0.04168006 nats. Its cases SHA256cbd2632af41a8c6ca8cbe55066d77b98791565b7ac301b0bea1965e84ec50f5b and bank SHA256703671b0fb01ddf74dfe60d51e44914c71647c92e7e5ee51d111a9acfcecfde8 exactly match V4's paired records. It is an archived same-bank reference, not a new independent V5 confirmation. Native vsOED seed102–105 counterparts are not verified by this task. V4 reference EIGs remain10.98247613/11.08123489; no new V5 EIG exists.

## Source preservation

All pre-existing source/config/tools files checked by the earlier796-file inventory remained unchanged except the two explicit additive dispatch/registry hooks. Stable remains unchanged. Full allowlist hashes, hook backups/diffs and test receipts accompany this addendum. No automatic sync, commit, push, stable promotion or long GPU campaign.
