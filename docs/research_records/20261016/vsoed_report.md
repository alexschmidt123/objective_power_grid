# vsOED: troubleshooting report

This is a development troubleshooting report, not final confirmatory evidence. Training seeds 101–105 are running under the frozen fixed-budget protocol; only completed methods enter this snapshot. Evaluation uses seed 1001, 2,000 paired trajectories, L=1,000,000, and the final 1M-budget checkpoint. No retraining, checkpoint reselection, algorithm change, or reserved-seed evaluation was performed for this audit.

A nonzero gradient (>1e-8) is an exposure proxy, not proof of informative learning. Raw gradient norms are measured before component clipping; large raw values are not the applied parameter update. Correlation with sharpening does not establish the router as the sole cause.

## Assessment

Seed sensitivity and incomplete convergence evidence; no catastrophic collapse found in inspected curves.

Seeds 102 and 103 improve through the final validation checkpoint. Seed 104 has a modest late decrease. The observed seed range is materially smaller than the ScoreBED/MoE failures, although larger than trajectory Monte Carlo standard errors. This is not proof that native actor–critic or variational posterior learning is numerically optimal.

The training reward is native variational reward and is not interchangeable with common held-out EIG. Incomplete critic/posterior loss records limit causal diagnosis; reward trends alone cannot isolate actor, critic or posterior approximation error. No extra primary episodes should be added to this method alone in the 1M study.

No rerun is currently justified. Report fixed-budget performance and unverified convergence. Only a demonstrated implementation/protocol defect would justify replacing the configuration, and affected seeds would need consistent treatment. Do not reuse the historical 5M result in the main comparison.

## Measured evidence

| Seed | EIG | Trajectory MC SE | Boundary coordinates | Final-budget selection | Checkpoint finite / hash matches |
|---|---:|---:|---:|---|---|
| 101 | 8.58218 | 0.04832 | 0.04% | final, as upstream | True / True |
| 102 | 8.75028 | 0.04729 | 0.02% | final_predeclared_fixed_budget | True / True |
| 103 | 7.96138 | 0.05066 | 0.19% | final_predeclared_fixed_budget | True / True |
| 104 | 8.42605 | 0.04968 | 0.14% | final_predeclared_fixed_budget | True / True |

Across 4 completed training seeds: mean 8.42997, sample SD 0.33930. This is an interim summary, not a five-seed result.

## Reproducibility and limitations

See `evidence.json` for run paths, complete stage usage, validation history, gradients and checkpoint checks; `source_hashes.json` hashes inputs. Only saved complete runs are audited. The previous disk-full partial MoE-103 attempt remains archived and is not the accepted checkpoint. External GPU sharing affects wall-clock comparisons; no evidence identifies it as the cause of EIG degradation. No final seed 8831001 was used.
