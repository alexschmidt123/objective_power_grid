# MoE-sBOED location-finding benchmark

Status: COMPLETE; 50/50 method-seed evaluations complete.

| Method | 101 | 102 | 103 | 104 | 105 | Mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| MoE-sBOED | 12.89948 | 12.95551 | 12.86055 | 12.94789 | 12.92164 | 12.91701 ± 0.03857 |
| MoE-sBOED-uniform | 12.88328 | 12.94091 | 12.85165 | 12.96172 | 12.93317 | 12.91415 ± 0.04528 |
| MoE-sBOED-E1 | 9.88789 | 9.48270 | 9.82469 | 9.81106 | 9.78657 | 9.75858 ± 0.15870 |
| MoE-sBOED-E2 | 11.50957 | 11.61950 | 11.57831 | 11.71277 | 11.57085 | 11.59820 ± 0.07513 |
| MoE-sBOED-E3 | 10.41429 | 10.31626 | 10.35975 | 10.31854 | 10.31404 | 10.34458 ± 0.04330 |
| MoE-sBOED-E4 | 11.33666 | 11.39588 | 11.42926 | 11.53348 | 11.44477 | 11.42801 ± 0.07208 |
| DAD | 10.31407 | 10.35344 | 9.55718 | 10.49226 | 8.53541 | 9.85047 ± 0.82086 |
| vsOED | 12.44272 | 12.30648 | 12.43011 | 12.26366 | 12.26530 | 12.34165 ± 0.08830 |
| RL-sBOED | 11.19165 | 11.17205 | 11.14941 | 11.12023 | 11.14663 | 11.15599 ± 0.02711 |
| ScoreBED | 10.99883 | 11.12651 | 10.91054 | 11.06600 | 11.07487 | 11.03535 ± 0.08325 |

E1–E4 are extracted standalone policies from the matching learned seed,not independently trained experts. MoE seeds are conditional on common pretraining. No superiority claim follows from a higher point estimate. Primary paired intervals and Holm-adjusted tests require all five seed pairs. Case MCSE and across-seed SD are separate in BENCHMARK_RESULTS.json.

The controller preserves uniform seed101 and excludes all15 development-ablation training jobs. New internal training uses only native variational EIG;native external methods retain their own legitimate objectives. Four-hour total cap includes each training run and its required evaluations.

Reproduce evaluation with the exact original bank seed202611099206,2000 truth/noise cases,1M contrasts,T30 and frozen scorer. Never substitute1001 or choose checkpoints with this bank. Source/config/checkpoint paths are in CHECKPOINT_INVENTORY.json;compute records are in COMPUTE_FAIRNESS.json.
