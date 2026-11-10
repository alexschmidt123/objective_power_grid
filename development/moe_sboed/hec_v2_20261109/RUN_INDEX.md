# HEC-MoE-sBOED v2

Implementation and pretraining validation. No new scientific training has started yet. This directory is LabPC's authoritative development source. The isolated runtime/output directory is `~/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109/`; read its RUN_INDEX.md and evidence receipts for live status after launch.

One candidate, one trained stage-only v2 control and one ordinary trained HEC-v1 continuation are authorized. All start from the verified successful HEC-v1 terminal checkpoint; none retrains v1 from fresh initialization. Each gets 2,000 native updates with the same saved replay/optimizer schedules where compatible. All original artifacts remain immutable. GitHub synchronization to Ubuntu is code-only; no Ubuntu computation is launched.
