# SwitchMoE-sBOED: troubleshooting report

This is a development troubleshooting report, not final confirmatory evidence. Training seeds 101–105 are running under the frozen fixed-budget protocol; only completed methods enter this snapshot. Evaluation uses seed 1001, 2,000 paired trajectories, L=1,000,000, and the final 1M-budget checkpoint. No retraining, checkpoint reselection, algorithm change, or reserved-seed evaluation was performed for this audit.

A nonzero gradient (>1e-8) is an exposure proxy, not proof of informative learning. Raw gradient norms are measured before component clipping; large raw values are not the applied parameter update. Correlation with sharpening does not establish the router as the sole cause.

## Assessment

Confirmed seed-103 optimization breakdown and failure to demonstrate learned hard continuation.

The same late sPCE deterioration and gradient instability occurs in seed 103. Seeds 101–104 hard routing statistics are measured from saved deployed rollouts, excluding the initial routed initialization from continue/reroute denominators. A rerouting request may keep the same expert; actual changes are reported separately. Long expert durations do not prove that the gate chose continuation.

The gate is initialized near 0.99 rerouting. This is a plausible contributor to absence of persistence, not a demonstrated cause of EIG collapse. Do not simultaneously change gate initialization and router sharpening. First isolate the shared MoE instability. Gate-calibration interventions should be a separate subsequent development experiment with unchanged experts and router configuration.

Journal claim restriction: current evidence supports the implemented switching architecture, but not useful learned temporal persistence or a switching-only performance advantage.

## Measured evidence

| Seed | EIG | Trajectory MC SE | Boundary coordinates | Final-budget selection | Checkpoint finite / hash matches |
|---|---:|---:|---:|---|---|
| 101 | 10.10045 | 0.04662 | 0.00% | final_predeclared_fixed_budget | True / True |
| 102 | 10.03829 | 0.04714 | 0.00% | final_predeclared_fixed_budget | True / True |
| 103 | 6.50948 | 0.06129 | 13.97% | final_predeclared_fixed_budget | True / True |
| 104 | 10.61479 | 0.04816 | 0.00% | final_predeclared_fixed_budget | True / True |

Across 4 completed training seeds: mean 9.31575, sample SD 1.88860. This is an interim summary, not a five-seed result.

| Seed | Final-fifth mean raw norm | Router p99 raw norm | Router max raw norm | Final batch sPCE | Expert usage E1/E2/E3/E4 |
|---|---:|---:|---:|---:|---|
| 101 | 7.69e+03 | 3.43e+04 | 9.96e+06 | 6.78777 | 13.24%/6.08%/80.68%/0.00% |
| 102 | 1.5e+03 | 6.38e+03 | 2.48e+06 | 6.80715 | 27.07%/0.21%/72.73%/0.00% |
| 103 | 1.1e+07 | 1.68e+08 | 4.7e+09 | 5.37358 | 62.95%/0.00%/37.05%/0.00% |
| 104 | 69.2 | 604 | 5.02e+03 | 6.89699 | 0.00%/96.24%/0.00%/3.76% |

Final-batch score quantiles are [minimum, p10, median, p90, p99, maximum]; they describe one batch, not the whole training distribution. Full traces contain per-update batch summaries.

- Seed 101: {'quantiles': [3.6232669364527865, 6.706177477023625, 6.932428314976152, 6.932447891572471, 6.932447891572508, 6.932447891572508], 'fraction_within_01_of_ceiling': 0.859375, 'fraction_within_05_of_ceiling': 0.9375}.

- Seed 102: {'quantiles': [4.161981306412848, 6.743017855422578, 6.932408362661546, 6.932447891572496, 6.932447891572509, 6.932447891572512], 'fraction_within_01_of_ceiling': 0.890625, 'fraction_within_05_of_ceiling': 0.9375}.

- Seed 103: {'quantiles': [-2.503812024474943, 3.6402241573510308, 5.493024939137198, 6.9319490231547025, 6.932447891572486, 6.932447891572508], 'fraction_within_01_of_ceiling': 0.25, 'fraction_within_05_of_ceiling': 0.328125}.

- Seed 104: {'quantiles': [6.00954163046886, 6.904704911533449, 6.932444308053437, 6.932447891572508, 6.932447891572508, 6.932447891572508], 'fraction_within_01_of_ceiling': 0.953125, 'fraction_within_05_of_ceiling': 0.96875}.

- Seed 101: deployed reroute requests 100.00%; actual changes 7.15%; mean expert run length 9.657.

- Seed 102: deployed reroute requests 100.00%; actual changes 4.73%; mean expert run length 12.476.

- Seed 103: deployed reroute requests 100.00%; actual changes 55.05%; mean expert run length 1.767.

- Seed 104: deployed reroute requests 100.00%; actual changes 3.23%; mean expert run length 15.239.

## Reproducibility and limitations

See `evidence.json` for run paths, complete stage usage, validation history, gradients and checkpoint checks; `source_hashes.json` hashes inputs. Only saved complete runs are audited. The previous disk-full partial MoE-103 attempt remains archived and is not the accepted checkpoint. External GPU sharing affects wall-clock comparisons; no evidence identifies it as the cause of EIG degradation. No final seed 8831001 was used.
