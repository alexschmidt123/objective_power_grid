# Validated HEC-MoE source publication

LabPC is authoritative. This is a byte-identical source/configuration snapshot of the validated running experiment; SOURCE_PROVENANCE.json records the correspondence. Updating this development copy does not authorize modifying a running or frozen experiment.

Checkpoints, datasets, evaluation banks, training outputs and credentials are intentionally absent. The source currently references historical native, Collaborative and SG snapshots and saved runtime state through the LabPC layout. Code synchronization alone does not provision a runnable Ubuntu training environment. An explicitly assigned worker computation must first receive approved dependencies, path mapping and verified model inputs separately. Do not silently replace missing provenance or launch a job during synchronization.

For new logical changes: validate on LabPC, review Git changes, commit/push this branch and synchronize the clean worker through tools/moe_sync.py from MacBook. Follow the root AGENTS.md and docs/MOE_SYNC_WORKFLOW.md. No force resets, automatic stashing, or overwriting dirty workers.
