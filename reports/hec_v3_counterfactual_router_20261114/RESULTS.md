# Frozen-expert V3 counterfactual router pilot

Completed router-only pilot. Experts and context remained unchanged; historical artifact hashes passed. This is one training lineage and does not establish multi-seed superiority.

| Policy | Prior-sPCE (nats) |
|---|---:|
| original | 13.83395320 |
| counterfactual | 13.83220028 |
| critic_control | 13.84259316 |
| uniform | 13.89225873 |
| expert1 | 13.71760341 |
| expert2 | 13.87451022 |
| expert3 | 13.92008640 |
| expert4 | 13.85662145 |
| V2 | 14.04681863 |

Evaluation: 1024 paired trajectories,16 million shared contrasts. This is a fresh bank; do not compare these absolute means directly with old banks.

| Counterfactual minus comparator | Difference | 95% paired CI | Holm p |
|---|---:|---|---:|
| original | -0.00175292 | [-0.01587722, +0.01237138] | 1 |
| critic_control | -0.01039288 | [-0.02669927, +0.00591351] | 1 |
| uniform | -0.06005845 | [-0.15643670, +0.03631980] | 1 |
| expert1 | +0.11459686 | [+0.01656587, +0.21262786] | 0.153994 |
| expert2 | -0.04230995 | [-0.14447022, +0.05985033] | 1 |
| expert3 | -0.08788612 | [-0.19525292, +0.01948069] | 0.651169 |
| expert4 | -0.02442117 | [-0.09109362, +0.04225128] | 1 |
| V2 | -0.21461836 | [-0.31156707, -0.11766964] | 0.000123091 |

All-comparator pilot acceptance: **False**.

## What changed

The counterfactual arm trained routing on a fitted ensemble of paired native variational suffix-return advantages. The control retained frozen original critic advantages. Both collected their own current-policy histories and spent equal suffix-simulation and estimator-fitting budgets. The control is not a jointly updated native critic/actor baseline. Neither arm retrained experts, posterior networks or the critic.

| Arm/round | Accepted router steps | KL | Audit selected minus current [95% CI] |
|---|---:|---:|---|
| counterfactual/1 | 50 | 0.000009 | -0.031319 [-0.110822, +0.048185] |
| counterfactual/2 | 50 | 0.000009 | +0.014529 [-0.043514, +0.072573] |
| counterfactual/3 | 50 | 0.000034 | +0.010800 [-0.036728, +0.058327] |
| critic_control/1 | 50 | 0.000024 | -0.031319 [-0.110822, +0.048185] |
| critic_control/2 | 50 | 0.000104 | -0.002981 [-0.051603, +0.045641] |
| critic_control/3 | 50 | 0.000024 | +0.016206 [-0.021766, +0.054178] |

The audit table tests ensemble-selected actions against the pre-update mixture using16 untouched histories per round and128 independent suffix draws each. These exploratory intervals are not corrected across rounds and do not establish a root cause.

## Costs and limitations

Runtime: 1451.4 seconds. Training/history/suffix simulator-observation counts: `{'counterfactual': 3417024, 'critic_control': 3417024}`. Posterior importance likelihood calculations and final evaluation are additional computation. Each arm may accept fewer updates because of the same KL stopping rule.

Counterfactual labels use an approximate importance posterior with ESS/max-weight checks; this is not exact posterior sampling. Training optimizes the unchanged native variational information return, while final evaluation uses finite-contrast prior-sPCE. Small history coverage, finite suffix draws, learned value approximation and one training lineage limit conclusions. No final-bank tuning, deployment-time optimization or expert retraining was performed.

Raw authoritative artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_counterfactual_router_20261114_pilot02`.

## Heldout value prediction

| Counterfactual round | Ensemble advantage MSE | Zero-advantage MSE | Original critic MSE |
|---|---:|---:|---:|
| counterfactual_round1 | 0.01206285 | 0.01171791 | 0.01125221 |
| counterfactual_round2 | 0.00485916 | 0.00467180 | 0.00489954 |
| counterfactual_round3 | 0.01116683 | 0.01130751 | 0.01169022 |

These16-history independent-suffix checks show whether the fitted estimator predicts advantages beyond a zero baseline. They do not establish that conditional routing headroom is absent. Each value network has91,588 parameters, with only64 fitting histories per round; validation early stopping limits fitting but cannot establish generalization. Posterior quality and empirical per-history selection diagnostics are in VALUE_DIAGNOSTICS.json.
