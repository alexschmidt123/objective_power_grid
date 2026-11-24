# Final location-finding benchmark

All50 method-seed evaluations are complete. Proposed method:MoE-sBOED;uniform is independently trained;E1–E4 are extracted standalone policies from corresponding learned checkpoints.

| Method | Mean EIG ± seed SD (nats) |
|---|---:|
| MoE-sBOED | 12.91701 ± 0.03857 |
| MoE-sBOED-uniform | 12.91415 ± 0.04528 |
| MoE-sBOED-E1 | 9.75858 ± 0.15870 |
| MoE-sBOED-E2 | 11.59820 ± 0.07513 |
| MoE-sBOED-E3 | 10.34458 ± 0.04330 |
| MoE-sBOED-E4 | 11.42801 ± 0.07208 |
| DAD | 9.85047 ± 0.82086 |
| vsOED | 12.34165 ± 0.08830 |
| RL-sBOED | 11.15599 ± 0.02711 |
| ScoreBED | 11.03535 ± 0.08325 |

| MoE-sBOED minus | Paired mean difference | 95% seed-level CI | Holm p |
|---|---:|---|---:|
| MoE-sBOED-uniform | +0.00287 | [-0.01510,+0.02083] | 0.68079 |
| MoE-sBOED-E1 | +3.15843 | [+2.92911,+3.38775] | 1.1172e-05 |
| MoE-sBOED-E2 | +1.31881 | [+1.24350,+1.39412] | 6.424e-06 |
| MoE-sBOED-E3 | +2.57244 | [+2.48102,+2.66385] | 1.2868e-06 |
| MoE-sBOED-E4 | +1.48900 | [+1.40235,+1.57565] | 6.424e-06 |
| DAD | +3.06654 | [+2.06199,+4.07109] | 0.0021244 |
| vsOED | +0.57536 | [+0.42468,+0.72604] | 0.0013441 |
| RL-sBOED | +1.76102 | [+1.69754,+1.82450] | 1.2868e-06 |
| ScoreBED | +1.88166 | [+1.82269,+1.94063] | 8.7581e-07 |

**Learned-routing advantage is not demonstrated**: paired difference+0.00287 nats,95% CI[-0.01510,+0.02083],Holm p=0.6808.

The other comparisons must be read with their paired intervals and adjusted tests above. Strongest fixed extracted expert is MoE-sBOED-E2. Its selection is covered by simultaneous expert comparisons,not a new unadjusted selected-expert test.

## Scope and fairness

These are five continuation seeds conditional on a shared pretrained initialization,not five independent end-to-end pretraining lineages. Learned/uniform pairs use matched initial weights,2000 actor updates,10000 critic updates,300000 posterior updates,2M new episodes and60M observations. All10 completed below the four-hour new-run cap. Extracted expert evaluations use the same trained parent and a validated one-current-expert interface with common inherited context;they are not independently optimized single-expert baselines.

External native training is reused. Historical RL-sBOED used approximately7h37m per seed;it is not relabeled as four-hour training. Native objectives,model capacity,inherited pretraining and simulation counts differ. This study is not a perfectly compute-matched or from-scratch superiority comparison.

All policies use the original fixed evaluation seed202611099206,2000 truth/base-noise cases,1M prior contrasts,T30 and identical frozen likelihood/physical bounds. Different policies generate different realized histories from the coupled randomness. The bank has informed prior development,so it is not an untouched final test. Terminal checkpoints were used;no new test-EIG selection or tuning. Case MCSE is reported separately from seed variability and conditions on the common contrast bank. Five-seed t intervals have small-sample assumptions.

## Artifacts

MAIN_RESULTS.csv contains every seed.EIG tables/statistics are in BENCHMARK_RESULTS.json;checkpoint/config provenance is in CHECKPOINT_INVENTORY.json;resource data are in COMPUTE_FAIRNESS.json and COMPUTE_FAIRNESS.csv.Figures:five_seed_comparison.png,learned_vs_uniform.png,expert_comparison.png.Raw checkpoints/logs/scores remain in the main project experiments directories;none are overwritten or retrained.
