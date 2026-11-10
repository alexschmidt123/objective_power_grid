# V3 preflight only

No training launcher is provided. Read RUN_INDEX.md. Scripts use explicit existing frozen sources and independently tested CPU fixtures;they never train a model. Outputs belong in the isolated scratch run,not theGitdirectory. Never infer completed v3 training from existing HECEIG tables.

Run gradient_preflight.py --output TEMP/scratch_runs/moe_sboed_v3_competitive_20261110/evidence using the existing dellma environment. frozen_history_audit.py refuses to overwrite its completed receipt and respects the existingGPUlease. write_dossier.py renders the actual receipts;it performs no scientific computation.
