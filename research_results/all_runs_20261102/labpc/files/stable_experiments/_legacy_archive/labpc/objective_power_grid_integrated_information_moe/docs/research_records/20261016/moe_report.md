# MoE-sBOED: troubleshooting report

This is a development troubleshooting report, not final confirmatory evidence. Training seeds 101–105 are running under the frozen fixed-budget protocol; only completed methods enter this snapshot. Evaluation uses seed 1001, 2,000 paired trajectories, L=1,000,000, and the final 1M-budget checkpoint. No retraining, checkpoint reselection, algorithm change, or reserved-seed evaluation was performed for this audit.

A nonzero gradient (>1e-8) is an exposure proxy, not proof of informative learning. Raw gradient norms are measured before component clipping; large raw values are not the applied parameter update. Correlation with sharpening does not establish the router as the sole cause.

## Assessment

Confirmed optimization breakdown in seed 103; root cause not fully isolated.

Seed 103 training sPCE falls during the last fifth while raw gradients explode. Its final held-out EIG is below Random's historical 8.26624 reference. The accepted run completes all 15,625 updates and 1M episodes; it is not a prematurely terminated model. Late relaxed-training deterioration means hard inference alone cannot explain every aspect of the failure. However, hard-versus-relaxed held-out EIG remains unmeasured in this audit.

The successful seed 104 despite highly concentrated routing shows that concentration alone is insufficient to explain failure. Assess starvation using exposure, norm distributions and proposal quality jointly.

Next diagnostic after current runs: paired hard/relaxed evaluations of the same good and failed checkpoints; locate the deterioration against gamma and group gradient spikes. If a correction is justified, change one sharpening/optimization factor only in a separate development protocol. Preserve expert widths, encoder, objective, training count and evaluator. Do not choose a new setting simply because it repairs seed 103.

## Measured evidence

| Seed | EIG | Trajectory MC SE | Boundary coordinates | Final-budget selection | Checkpoint finite / hash matches |
|---|---:|---:|---:|---|---|
| 101 | 9.82944 | 0.04801 | 0.00% | final | True / True |
| 102 | 9.72441 | 0.04981 | 0.00% | final_predeclared_fixed_budget | True / True |
| 103 | 6.25322 | 0.05495 | 0.00% | final_predeclared_fixed_budget | True / True |
| 104 | 10.41524 | 0.04942 | 0.00% | final_predeclared_fixed_budget | True / True |

Across 4 completed training seeds: mean 9.05558, sample SD 1.89280. This is an interim summary, not a five-seed result.

| Seed | Final-fifth mean raw norm | Router p99 raw norm | Router max raw norm | Final batch sPCE | Expert usage E1/E2/E3/E4 |
|---|---:|---:|---:|---:|---|
| 101 | 5.07e+04 | 4.34e+04 | 7.31e+07 | 6.82577 | 0.23%/26.46%/57.11%/16.21% |
| 102 | 9.54e+03 | 5.01e+04 | 8.34e+06 | 6.88181 | 19.21%/0.50%/80.29%/0.00% |
| 103 | 1.09e+07 | 1.39e+08 | 1.12e+10 | 5.67535 | 45.46%/0.13%/42.27%/12.13% |
| 104 | 396 | 1.98e+03 | 5.95e+05 | 6.80709 | 0.09%/89.28%/0.08%/10.55% |

Final-batch score quantiles are [minimum, p10, median, p90, p99, maximum]; they describe one batch, not the whole training distribution. Full traces contain per-update batch summaries.

- Seed 101: {'quantiles': [5.491303979149574, 6.743726141621784, 6.932347798575771, 6.932447891485415, 6.932447891572509, 6.932447891572512], 'fraction_within_01_of_ceiling': 0.84375, 'fraction_within_05_of_ceiling': 0.9375}.

- Seed 102: {'quantiles': [6.2076695055035955, 6.833113334856499, 6.93244026259501, 6.93244789157243, 6.932447891572508, 6.932447891572508], 'fraction_within_01_of_ceiling': 0.890625, 'fraction_within_05_of_ceiling': 0.96875}.

- Seed 103: {'quantiles': [-0.1424843850947788, 3.6421181704921866, 6.328269104684697, 6.931930141775418, 6.932447891569159, 6.932447891571737], 'fraction_within_01_of_ceiling': 0.296875, 'fraction_within_05_of_ceiling': 0.484375}.

- Seed 104: {'quantiles': [2.953937003694417, 6.87867807185903, 6.932374476528457, 6.932447891572508, 6.932447891572508, 6.932447891572508], 'fraction_within_01_of_ceiling': 0.921875, 'fraction_within_05_of_ceiling': 0.9375}.

## Reproducibility and limitations

See `evidence.json` for run paths, complete stage usage, validation history, gradients and checkpoint checks; `source_hashes.json` hashes inputs. Only saved complete runs are audited. The previous disk-full partial MoE-103 attempt remains archived and is not the accepted checkpoint. External GPU sharing affects wall-clock comparisons; no evidence identifies it as the cause of EIG degradation. No final seed 8831001 was used.
