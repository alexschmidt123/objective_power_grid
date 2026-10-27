# Experiment storage policy

Only ~/Documents/objective_power_grid (STABLE) and ~/Documents/objective_power_grid_temp (TEMP) are active project roots on each PC. LabPC STABLE is authoritative for integrated publication results. No additional top-level project copies may be created.

Development code belongs in TEMP. New development runs initially write only to TEMP/scratch_runs/. TEMP must not contain a permanent experiments/ directory. Historical source snapshots and working patches are under TEMP/development_sources/. Binary test fixtures are source regression fixtures, not scientific results. Scientific datasets and historical reference checkpoints are preserved in STABLE.

Permanent official results belong only in STABLE/experiments/location_finding/ or STABLE/experiments/sir_ode/. Use a new, nonconflicting run identifier. STABLE/experiments/RESULT_REGISTRY.json explicitly controls official inclusion; never recursively collect _legacy_archive/ into benchmark tables. Old completion flags do not confer current validation.

Promotion requires, in order:
1. Verify mathematics and numerical correctness with independent tests.
2. Verify frozen source revision, effective configuration, training seed, checkpoint and SHA256, and scientific provenance.
3. Verify raw EIG scores, summary, statistical uncertainty, and paired evaluation identities.
4. Verify the declared training/evaluation protocol, complete budgets, logs, and compute costs.
5. Record a scientific promotion decision. Valid negative evidence may be NEGATIVE; never label it a successful method. Failed/incomplete evidence stays unvalidated.
6. Integrate verified TEMP source through controlled Git integration on LabPC and pass relevant regression tests in STABLE.
7. Copy artifacts into a new STABLE benchmark directory without overwriting an existing run.
8. Independently verify destination SHA256 values and metadata.
9. Register the result explicitly in STABLE/experiments/RESULT_REGISTRY.json and reports/EXPERIMENT_PROMOTION_REGISTRY.md.
10. Only then remove the redundant TEMP output. Never delete the sole surviving checkpoint or dataset.

Failed or incomplete development outputs remain in TEMP/scratch_runs/ pending review with source/configuration hashes and useful failure evidence. If permanent preservation is necessary, archive under STABLE/experiments/_legacy_archive/ and label UNVALIDATED, FAILED, INCOMPLETE, or EXCLUDED_LEGACY. Archive entries are excluded automatically from current tables. Archived duplicate files can share verified read-only inodes; never edit archived files in place. Copy an artifact to a fresh development path before modification.

Official training uses frozen validated STABLE code. The maintained src.online_cli entrypoint enforces TEMP scratch output boundaries, rejects existing output identifiers, and requires an exact-command official-run authorization with a complete source/configuration SHA256 manifest and passing regression status for STABLE. These checks do not establish scientific validity or grant permission to launch an experiment. Internal Python workers must not bypass the root run.sh/sweep_run.sh workflow. Historical launchers are archival and must not be resumed automatically.

STABLE authorization records live in reports/official_run_authorizations/<SHA256-of-compact-JSON-argv>.json. Required fields are status=VALIDATED_FROZEN_SOURCE, regression_status=PASS, argv (exact argument list), and source_sha256 (all .py/.yaml/.yml/.sh files under src/tools/configs/scripts plus run.sh and sweep_run.sh). No live authorization is created by consolidation.

UbuntuPC-to-LabPC promotion must preserve originating-machine metadata, transfer with independent SHA256 checks, reject conflicting run identifiers, and register in LabPC's authoritative registry only after validation. Only LabPC may commit or push to GitHub. Checkpoints and raw datasets must not enter Git. This consolidation does not launch GPU experiments or change unrelated projects.

Historical saved commands/configuration files remain byte-identical. Resolve old paths through reports/PROJECT_PATH_RELOCATION.json; recover source snapshots from the recorded verified tar archives and pinned Git refs. Source archives are not executable aliases of removed checkouts.
