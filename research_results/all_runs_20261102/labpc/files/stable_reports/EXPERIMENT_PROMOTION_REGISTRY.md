# Experiment promotion registry

No scientific experiment was promoted by the storage consolidation. Scientific validation gates have not been rerun for historical campaigns; existing evidence is preserved without reclassification as a current successful method. Both benchmark directories are reserved for fully validated future promotions.

The canonical machine-readable registry is experiments/RESULT_REGISTRY.json. All preserved historical entries are EXCLUDED_LEGACY, except the interrupted corrected seed101 SIR journal run, which is INCOMPLETE. Archive inclusion never implies success or scientific invalidity: it records that the new promotion checklist has not been completed.

The only source integration is the tested storage-boundary guard and CLI dispatch. Its CPU regression commands, outputs, and source hashes are in reports/STORAGE_POLICY_REGRESSION.json. No algorithmic development changes were promoted. Original working changes are preserved in TEMP, and old worktree revisions are pinned under refs/archive/consolidation-20261027/ with a verified shared Git bundle.

Future entries must identify benchmark, immutable run ID, originating machine, source revision and hashes, effective config and seed, checkpoint SHA256, logs and compute costs, raw scores and uncertainty, protocol/correctness evidence, Git integration and regression receipt, destination verification receipt, status (VALIDATED or NEGATIVE), and explicit table-inclusion decision.


## SIR comparison rejected and stopped

Campaign `10272026_234851_corrected_baselines_seed101_fourhour`: **EXCLUDED_UNDERTRAINED**, not accepted as valid benchmark results. Completed and partial records preserved. Remaining RL stopped; vsOED and ScoreBED cancelled. No automatic restart or promotion. Mathematical source validation is separate from adequate training; the current results do not establish adequately trained method performance.
