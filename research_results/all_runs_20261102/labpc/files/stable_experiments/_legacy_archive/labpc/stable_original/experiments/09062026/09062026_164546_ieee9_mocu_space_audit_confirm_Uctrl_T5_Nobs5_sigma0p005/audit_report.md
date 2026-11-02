# Finite-loss MOCU space audit

Phase: confirm. 64 validation systems × 3 observation-noise seeds; no policy training.
Posterior support: 384; actions: 54; prior Bayes risk: 0.028958.

Positive gains favor the adaptive policy in each comparison. Intervals cluster repeated seeds by validation system.
Metric: candidate-grid bank regret u + 20 max(U_grid-u, 0) - U_grid.
This uses bank U_grid, not the continuous bisection oracle used in final realized operational regret.
This is a diagnostic, not a final performance table or physical-safety certification.

## Horizon 2

- Myopic adaptation over calibrated Fixed: 0.003438; paired 95% interval [0.000313, 0.007083].
- Combined lookahead gain over calibrated Fixed: -0.000990; paired 95% interval [-0.005677, 0.003490].
- Non-myopic gain over Myopic: -0.004427; paired 95% interval [-0.010522, 0.000417].
- Status: `material_gain_not_shown_by_this_planner`.
- Lookahead bank-based coverage: 0.969.

## Horizon 3

- Myopic adaptation over calibrated Fixed: -0.003333; paired 95% interval [-0.008178, 0.001406].
- Combined lookahead gain over calibrated Fixed: -0.003906; paired 95% interval [-0.010156, 0.001615].
- Non-myopic gain over Myopic: -0.000573; paired 95% interval [-0.007760, 0.005833].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.974.

## Horizon 4

- Myopic adaptation over calibrated Fixed: -0.002031; paired 95% interval [-0.007086, 0.002866].
- Combined lookahead gain over calibrated Fixed: -0.003490; paired 95% interval [-0.009948, 0.003802].
- Non-myopic gain over Myopic: -0.001458; paired 95% interval [-0.008125, 0.006042].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.958.

## Horizon 5

- Myopic adaptation over calibrated Fixed: 0.000781; paired 95% interval [-0.005938, 0.006979].
- Combined lookahead gain over calibrated Fixed: 0.001719; paired 95% interval [-0.002710, 0.006460].
- Non-myopic gain over Myopic: 0.000937; paired 95% interval [-0.003855, 0.005678].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.969.

## Limits

A weak approximate planner cannot prove there is no adaptive or non-myopic space.
Screen and confirm use disjoint validation halves. Freeze settings before confirmation; final held-out tests remain separate.
A positive planning signal does not guarantee DAD can learn it. More distinct action sequences alone do not establish valuable adaptivity.
Fixed is approximate; a stronger Fixed calibration may reduce the apparent adaptive advantage.
Intervals are unadjusted per comparison and conditional on the banks, planning seeds and Fixed calibration.
Confirmation halves are consumed after inspection; further tuning requires fresh validation for confirmatory claims.
The prior Bayes risk is an in-model perfect-information ceiling, not a bound on off-support empirical gains.
