# HEC-MoE v2 development

LabPC is authoritative. Source is validated here, committed/pushed, then synchronized through GitHub to the clean Ubuntu worker. No worker computation launches automatically.

The isolated execution directory is `~/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109/`. A validated immutable source copy and protocol manifest are made there before any preparation. Launch only its `source/launch_v2.py` once. The launcher sets MOE_V2_RUN_ROOT; development validation scripts require that variable explicitly. Do not edit locked execution sources or old experiments.

Files copied unchanged from v1 are identified by SOURCE_PROVENANCE.json for checkpoint-class compatibility. Main entrypoints are train_v2.py, prepare_v2.py and evaluate_v2.py, not the legacy compatibility modules. Historical native source snapshots, runtime checkpoints and evaluation dependencies remain outside Git; synchronization alone is not a provisioned Ubuntu training environment. Future explicitly assigned worker computation requires verified inputs/path mapping separately. No private data, checkpoints or large results belong in Git.
