# Replacement V3 pilot results

Accepted against all predeclared comparators: **False**.

Exact original V2 evaluation:2,000 cases/1M contrasts,frozen scorer and physics. No comparison uses the previous16M bank.

| Policy | Estimated EIG(nats) |
|---|---:|
| V2 | 12.92935261 |
| V2_uniform | 12.93794410 |
| vsoed | 12.44271971 |
| dad | 10.31407383 |
| rl_sboed | 11.19164733 |
| scorebed | 10.99883133 |
| retired_original_V3 | 12.80310089 |
| retired_counterfactual | 12.80935983 |
| learned | 12.91092652 |
| uniform | 12.91284680 |
| expert1 | 11.25440346 |
| expert2 | 11.86222929 |
| expert3 | 11.40679259 |
| expert4 | 12.35935726 |
| control | 12.89702343 |
| V2_reproduction | 12.92935261 |

| Learned minus comparator | Difference | Paired95% CI | Holm p |
|---|---:|---|---:|
| V2 | -0.01842609 | [-0.05589997,+0.01904780] | 1 |
| V2_uniform | -0.02701758 | [-0.08228635,+0.02825119] | 1 |
| vsoed | +0.46820681 | [+0.37349501,+0.56291861] | 6.62492e-21 |
| dad | +2.59685269 | [+2.49176491,+2.70194047] | 0 |
| rl_sboed | +1.71927919 | [+1.62247435,+1.81608404] | 4.56345e-207 |
| scorebed | +1.91209519 | [+1.80912796,+2.01506242] | 4.53516e-222 |
| retired_original_V3 | +0.10782563 | [+0.05963856,+0.15601270] | 7.20944e-05 |
| retired_counterfactual | +0.10156668 | [+0.05357337,+0.14956000] | 0.00017297 |
| uniform | -0.00192029 | [-0.05672510,+0.05288453] | 1 |
| expert1 | +1.65652306 | [+1.55892586,+1.75412026] | 1.20975e-192 |
| expert2 | +1.04869723 | [+0.95571863,+1.14167583] | 3.30593e-96 |
| expert3 | +1.50413393 | [+1.40950856,+1.59875930] | 3.29635e-173 |
| expert4 | +0.55156926 | [+0.47221378,+0.63092474] | 1.27617e-39 |
| control | +0.01390309 | [-0.03273887,+0.06054505] | 1 |

Short one-lineage pilot,internal standalone ablations not independently optimized baselines;native-update/simulator budgets matched to control,parameter counts and FLOPs differ;1M prior-sPCE ceiling limited.

Byte-verified exact saved reference vectors on unchanged bank/scorer/checkpoints;V2 additionally rerolled and rescored.

Raw artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_feature_20261114_pilot`.
