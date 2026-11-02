# Blocker and staged integration requirements

The authoritative V4 utility-routing/formal-training modules are absent from both active Ubuntu roots. No V4-named source/config/checkpoint artifacts were found by the recorded recursive inventory. Local Git also lacks LabPC source commit763a28dc38401fb9ac63f52bc2751338624c2b98. Existing V1 code shape alone is not adequate provenance.

## Required source/data package from LabPC, prepared read-only without disrupting its baseline campaign

1. The exact frozen source snapshot/manifest for TEMP/scratch_runs/moe_sboed_v4_formal_seeds101_102/, including DeepSeekPolicy and all imported V4 utility-routing/formal trainer/evaluator dependencies. Do not overlay Ubuntu's dirty source tree blindly.
2. Full seed101/102 final valid checkpoint files and their SHA256 receipts, effective configs, training manifests, compute ledgers and source hash manifests. Expected hashes from earlier conversation, NOT VERIFIED locally: seed101 d4b8c7fe3f80d650832e556fbbd112b4901b34d01b8fed96ec737db4ff22d5de; seed102 d470f54959a0b91e2708d429bf06474777d1c9d85739f572010cdcf69f518860.
3. Existing paired V1/V4 evaluation identities, results and benchmark definition; the matching native-vsOED location reference and its evaluation identity. Do not generate/open reserved seed8831001 data.
4. Non-overwriting staging destination: a fresh incoming/ subdirectory inside this audit's TEMP scratch folder, never a third checkout. Independently verify every transferred SHA256; inspect dependencies and protect Ubuntu SIR source before any selected integration into TEMP. Preserve all original paths/checkpoints.
5. Verify strict checkpoint loading,8434 parameters,one shared/seven routed/top3, matching source/config/horizon and exact V4-equivalent actions/loss/gradients with all V5 switches off. Only then implement behavior-preserving diagnostics. No stable promotion or full GPU campaign.

No transfer or source integration was executed by this audit. No runnable V5 launcher is claimed. A future integration command must use the actual verified package, not placeholder source files or a reconstructed four-expert policy.

## Commands actually executed

`ssh ubuntu-pc` for read-only inventory and environment inspection; project Python: `/home/qiqi/miniconda3/envs/mocu_optimized/bin/python`; preflight script is preserved beside this report. For a non-mutating status check:

```bash
ssh ubuntu-pc 'cd /home/qiqi/Documents/objective_power_grid_temp && git status --short && nvidia-smi'
```
