# Reproducing the V4 all-method comparison

V4 is the current moe-sboed. All ten method rows use the original canonical bank and frozen scorer, T=30, 2,000 cases and 1M contrasts. Original V4 pilot and replication banks remain separate. See PROTOCOL.json for exact hashes and the historical zero-error reproduction certificate.

The terminal V4 checkpoints are frozen. No training, tuning, reinitialization or checkpoint selection occurs. E1-E4 are executable single-specialist policies extracted from the corresponding learned checkpoint, with inherited common query provider, action head and weak correction. Neither the router nor the other specialists is evaluated. The decoder was trained on mixtures, so these are extracted-policy diagnostics, not independently optimized standalone methods.

CPU smoke: `CUDA_VISIBLE_DEVICES= python tools/moe_sboed_v4_allmethods_20261117.py --task smoke`.
GPU evaluation: `python tools/moe_sboed_v4_allmethods_20261117.py --task controller`, with existing resource locks. Completed jobs are skipped; preserved partial jobs require review. Use the original conda dellma environment. To regenerate tables without GPU evaluation use `--task report`. For independent CPU verification and scoped publication use `python tools/moe_sboed_v4_allmethods_publish.py`.

Raw score vectors, extracted state dictionaries, runtime logs and resource samples remain outside Git under experiments/location_finding/moe_sboed_v4_allmethods_20261117. CHECKPOINT_MANIFEST.json records original terminal checkpoint hashes and exact per-case score paths. External baselines reuse the verified historical native checkpoints and canonical score vectors.

Timing is CUDA-synchronized forward time including shared belief/model queries, at batch size 16. Amortized design and episode times are not single-case latency. Scoring and complete evaluation wall time are separate. GPU/CPU resource samples cover each six-policy worker; peak memory includes resident learned/uniform models. Historical baseline training resource measurements are not invented. COMPUTE_FAIRNESS.json preserves complete available training receipts and original V4 initialization/pretraining lineage. Extracted training cost is inherited, not zero total cost.

Five-seed t intervals use five paired seed differences. Case MCSE is separate and conditional on the common contrast bank. Multiple primary comparisons include Holm p values and simultaneous Bonferroni intervals. The strongest extracted expert must not be reported using an unadjusted selected-winner test. Shared pretraining and previous use of the canonical bank make this an exploratory continuation-seed comparison, not independent from-scratch or untouched-bank confirmation. Native baseline budgets differ; no perfect compute-matching claim.
