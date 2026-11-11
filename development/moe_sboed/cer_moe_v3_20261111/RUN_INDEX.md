# CER V3 run index

Authoritative Git source: TEMP/development/hec_git_sync/development/moe_sboed/cer_moe_v3_20261111.
All paths below are on LabPC, with TEMP=~/Documents/objective_power_grid_temp.

|Scratch directory under TEMP/scratch_runs|Role|State|
|---|---|---|
|cer_moe_sboed_v3_20261111|First fixed-probe audit|INVALIDATED; original bytes retained|
|cer_moe_sboed_v3_20261111_validated_values|Native-tail-corrected fixed-probe audit|Complete; four groups unsupported|
|cer_moe_sboed_v3_20261111_local_values|Fresh local-probe diagnostic|Complete; four groups unsupported|

Each valid scratch directory contains source/configuration snapshots,
evidence/IMPLEMENTATION_CHECKS.json, evidence/GROUP_SUPPORT.json,
evidence/signatures.npz and audit.log. The local diagnostic also contains
AUDIT_MANIFEST.json and audit_resources.jsonl. All raw numerical evidence stays
outside Git. The first audit's AUDIT_INVALIDATED.json points to its correction.

Use the existing /home/grads/g/g.lin/miniconda3/envs/dellma/bin/python environment.
Never rerun scientific audits into these completed directories. Value-signature
discovery refuses an existing GROUP_SUPPORT.json. CER_ROOT must name an explicit
isolated destination containing the applicable configuration/source snapshot.
No training launcher exists because the required group-support check failed.

Aggregate report reproduction (read-only inputs; writes only aggregate report
files to the specified destination):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/grads/g/g.lin/miniconda3/envs/dellma/bin/python \
  development/moe_sboed/cer_moe_v3_20261111/source/write_report.py \
  development/moe_sboed/cer_moe_v3_20261111
```

Unit/policy checks use source/checks.py with CER_ROOT set explicitly. They read
historical dependencies and run on CPU; they do not train a scientific policy.
The numerical group-discovery scripts use GPU leases and refuse concurrent
compute on an occupied GPU. No historical queue is resumed or modified.

Initial source commit9df38f4b; terminal-density correctionfc6a6938; fresh local
diagnostic source48a66ae8. Later report-only/defensive destination changes do not
replace the frozen numerical audit snapshots. Source is available on Ubuntu by
Git synchronization; the historical frozen dependencies are LabPC-local, so a
synced source tree alone does not provision a runnable Ubuntu experiment.
