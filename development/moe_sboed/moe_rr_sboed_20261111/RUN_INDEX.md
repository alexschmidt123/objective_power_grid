# RR-MoE pilot index

Authoritative source: TEMP/development/hec_git_sync/development/moe_sboed/moe_rr_sboed_20261111 on LabPC.
Scientific snapshot/output: TEMP/scratch_runs/moe_rr_sboed_20261111.
Disposable validation: TEMP/scratch_runs/moe_rr_validation_20261111.
TEMP is ~/Documents/objective_power_grid_temp on LabPC.

This is ONE new residual-routed candidate and ONE matched uniform-Top2 control.
The common preparation is followed sequentially by full then control, with no
restart, architecture sweep, extra seed or post-hoc extension. All previous
HEC,SG,V3 and V2.1 source/checkpoints remain untouched. Git contains safe source,
configuration,aggregate reports and provenance; model/runtime files,raw histories,
contrast banks and logs remain outside Git.

Frozen dependencies: scratch_runs/moe_hec_v2_icml_20261109 plus its preserved
SG/HEC/native source lineage. common.py verifies successful HEC-v2 iteration16001
actor/runtime hashes. Base construction changes only coordination to uniform
four-way feature weights. No separate trained uniform checkpoint is invented.

Use the existing /home/grads/g/g.lin/miniconda3/envs/dellma/bin/python for the
native pipeline and mocu_optimized for preserved external-baseline workers.
Set RR_ROOT explicitly. checks.py,physics_checks.py,preparation_checks.py and
evaluation_checks.py produce disposable receipts; lock.py requires successful
receipts and verified GitHub synchronization before any scientific update.
launch.py is duplicate-safe; driver.py queues on the existing GPU lease when busy.

Outputs include preparation/{actor.pt,runtime.pkl,RESULT.json},
arms/{full,control}/{actor.pt,runtime.pkl,RESULT.json}, periodic actor snapshots,
TRAINING_LOGS/,evaluation/,PRETRAIN_MANIFEST.json and ARTIFACT_SHA256.json.
All specified Markdown reports are generated from actual receipts. No incomplete
phase is relabeled as complete. MacBook completion watching publishes only
aggregate reports, then runs the approved LabPC→GitHub→Ubuntu synchronization.

No Ubuntu computation is launched. A synchronized tree does not provision
LabPC-local dependencies or authorize extra experiments.
