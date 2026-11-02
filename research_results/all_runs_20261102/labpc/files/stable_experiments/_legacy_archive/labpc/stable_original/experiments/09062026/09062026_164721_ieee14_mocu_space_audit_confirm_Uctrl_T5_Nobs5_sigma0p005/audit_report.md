# Finite-loss MOCU space audit

Phase: confirm. 64 validation systems × 3 observation-noise seeds; no policy training.
Posterior support: 384; actions: 84; prior Bayes risk: 0.023307.

Positive gains favor the adaptive policy in each comparison. Intervals cluster repeated seeds by validation system.
Metric: candidate-grid bank regret u + 20 max(U_grid-u, 0) - U_grid.
This uses bank U_grid, not the continuous bisection oracle used in final realized operational regret.
This is a diagnostic, not a final performance table or physical-safety certification.

## Horizon 2

- Myopic adaptation over calibrated Fixed: 0.000938; paired 95% interval [-0.000729, 0.003385].
- Combined lookahead gain over calibrated Fixed: -0.001979; paired 95% interval [-0.006146, 0.001824].
- Non-myopic gain over Myopic: -0.002917; paired 95% interval [-0.007083, 0.000208].
- Status: `material_gain_not_shown_by_this_planner`.
- Lookahead bank-based coverage: 0.984.

## Horizon 3

- Myopic adaptation over calibrated Fixed: 0.002240; paired 95% interval [-0.001354, 0.006095].
- Combined lookahead gain over calibrated Fixed: -0.000677; paired 95% interval [-0.006147, 0.004327].
- Non-myopic gain over Myopic: -0.002917; paired 95% interval [-0.007553, 0.001043].
- Status: `material_gain_not_shown_by_this_planner`.
- Lookahead bank-based coverage: 0.974.

## Horizon 4

- Myopic adaptation over calibrated Fixed: 0.001302; paired 95% interval [-0.001824, 0.005104].
- Combined lookahead gain over calibrated Fixed: -0.000990; paired 95% interval [-0.004480, 0.002397].
- Non-myopic gain over Myopic: -0.002292; paired 95% interval [-0.006199, 0.001458].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.984.

## Horizon 5

- Myopic adaptation over calibrated Fixed: 0.002292; paired 95% interval [-0.000781, 0.005833].
- Combined lookahead gain over calibrated Fixed: 0.001719; paired 95% interval [-0.002137, 0.005628].
- Non-myopic gain over Myopic: -0.000573; paired 95% interval [-0.003021, 0.001042].
- Status: `material_gain_not_shown_by_this_planner`.
- Lookahead bank-based coverage: 0.995.

## Limits

A weak approximate planner cannot prove there is no adaptive or non-myopic space.
Screen and confirm use disjoint validation halves. Freeze settings before confirmation; final held-out tests remain separate.
A positive planning signal does not guarantee DAD can learn it. More distinct action sequences alone do not establish valuable adaptivity.
Fixed is approximate; a stronger Fixed calibration may reduce the apparent adaptive advantage.
Intervals are unadjusted per comparison and conditional on the banks, planning seeds and Fixed calibration.
Confirmation halves are consumed after inspection; further tuning requires fresh validation for confirmatory claims.
The prior Bayes risk is an in-model perfect-information ceiling, not a bound on off-support empirical gains.
