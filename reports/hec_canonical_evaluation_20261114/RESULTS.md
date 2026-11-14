# Mandatory original-V2 evaluation: corrected V3 router comparison

This is the primary comparable result. The preceding 1,024-case/16M-contrast experiment is supplementary only and must not replace or be compared numerically with this benchmark.

All policies use the byte-identical saved V2 bank: 2,000 truth/noise cases, 1,000,000 prior contrasts, original routing uniforms, T=30, sigma=0.5, design bounds [-4,4], and the original float64 scorer (chunk 16,384, positive truth included once). No models were trained or selected by this rerun. Training budgets and checkpoint lineages remain different. Historical references are not newly promoted official benchmark results.

| Policy | Estimated EIG / prior-sPCE (nats) |
|---|---:|
| V2 | 12.92935261 |
| V2_uniform | 12.93794410 |
| vsoed | 12.44271971 |
| original_V3 | 12.80310089 |
| counterfactual | 12.80935983 |
| critic_control | 12.80030489 |
| uniform | 12.80527233 |
| expert1 | 12.73824955 |
| expert2 | 12.85301397 |
| expert3 | 12.86149725 |
| expert4 | 12.81226900 |
| dad | 10.31407383 |
| rl_sboed | 11.19164733 |
| scorebed | 10.99883133 |

| New router minus comparator | Difference | Paired 95% CI | Holm p |
|---|---:|---|---:|
| V2 | -0.11999277 | [-0.17022126,-0.06976428] | 2.68844e-05 |
| V2_uniform | -0.12858426 | [-0.18588278,-0.07128575] | 9.07269e-05 |
| vsoed | +0.36664012 | [+0.26929190,+0.46398835] | 2.20641e-12 |
| original_V3 | +0.00625895 | [-0.00227188,+0.01478977] | 0.498301 |
| critic_control | +0.00905494 | [-0.00152144,+0.01963132] | 0.466508 |
| uniform | +0.00408750 | [-0.04546245,+0.05363745] | 1 |
| expert1 | +0.07111029 | [+0.01899604,+0.12322454] | 0.0525795 |
| expert2 | -0.04365413 | [-0.09937338,+0.01206512] | 0.498301 |
| expert3 | -0.05213741 | [-0.10862121,+0.00434638] | 0.42245 |
| expert4 | -0.00290916 | [-0.04170475,+0.03588643] | 1 |
| dad | +2.49528601 | [+2.38962825,+2.60094376] | 0 |
| rl_sboed | +1.61771251 | [+1.51848741,+1.71693761] | 1.71875e-180 |
| scorebed | +1.81052851 | [+1.70915469,+1.91190232] | 6.5556e-209 |

## Historical reproduction checks

Each saved baseline score vector was reproduced over all 2,000 cases; acceptance tolerance is 1e-9 maximum absolute error.

| Reference | Maximum absolute score error |
|---|---:|
| V2 | 0 |
| V2_uniform | 0 |
| vsoed | 0 |
| dad | 0 |
| rl_sboed | 0 |
| scorebed | 0 |

## Protocol identity

- cases: `2000`
- contrasts: `1000000`
- horizon: `30`
- sigma: `0.5`
- chunk: `16384`
- bank_sha256: `52c0c614c4ee3b63d58ccda60714b38b00a2c0ca35564a428fe8cc2ce8d727c5`
- uniforms_sha256: `6e56f0a08fe4deeefd7f0977967dba9d2910a4a383f1acfbd1736db3f358033c`
- scorer_sha256: `64c67b6547c77f4823b487493ed0703efbd3f04d7308019e875d145505695f2f`

Elapsed evaluation time: 567.3 seconds; training updates: 0. Historical artifacts and evaluated model states were unchanged.

The 1M estimator is ceiling limited and this is one training lineage. Paired intervals condition on this fixed contrast bank; they do not remove finite-contrast bias or establish multi-seed robustness. The rule prevents silent protocol substitutions; supplementary studies remain separately labeled.

Authoritative raw artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_canonical_evaluation_20261114`.

Rule: `AGENTS.md` and `development/moe_sboed/hec_canonical_evaluation_20261114/RULE.md`.
