# V4 promoted to moe-sboed: complete canonical-bank evaluation

User selected V4 as the proposed method. The original V4 pilot and new-bank replication remain untouched. This extension evaluates all ten V4 learned/uniform terminal checkpoints plus twenty extracted V4 specialist policies on the original canonical V2 bank. Twenty compatible external-baseline evaluations are reused with checkpoint hashes and protocol verification. No training, tuning, checkpoint selection, SIR or power-grid runs are authorized by this runner.

The extracted interface owns one copied specialist, common query/inference provider, inherited MoE action head and weak correction. It has no router or other experts and generates complete observed-history episodes. The shared head was trained on mixtures; extracted competence is not independent specialist training.

Run CPU audit: `CUDA_VISIBLE_DEVICES= python tools/moe_sboed_v4_allmethods_20261117.py --task smoke`. Run the sequential GPU-locked evaluation: `python tools/moe_sboed_v4_allmethods_20261117.py --task controller`. Existing completed rows are preserved; partial jobs require review. Regenerate tables with `--task report`.

Reports: RESULTS.md, RESULTS.json, MAIN_RESULTS.csv and CHECKPOINT_MANIFEST.json in this directory. Per-case arrays, extracted weights, logs and resource samples stay under experiments/location_finding/moe_sboed_v4_allmethods_20261117, outside Git.

The canonical bank informed earlier development. This is an exploratory comparison conditional on shared pretraining, not independent end-to-end or perfectly compute-matched superiority. Evaluation costs are additional post-training accounting, not hidden in the original four-hour training-run receipts.
