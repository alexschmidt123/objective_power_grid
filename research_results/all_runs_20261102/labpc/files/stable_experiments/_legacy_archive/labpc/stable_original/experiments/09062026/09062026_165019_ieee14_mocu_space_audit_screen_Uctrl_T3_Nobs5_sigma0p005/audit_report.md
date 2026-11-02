# Finite-loss MOCU space audit

Phase: screen. 64 validation systems × 3 observation-noise seeds; no policy training.
Posterior support: 384; actions: 84; prior Bayes risk: 0.023307.

Positive gains favor the adaptive policy in each comparison. Intervals cluster repeated seeds by validation system.
Metric: candidate-grid bank regret u + 20 max(U_grid-u, 0) - U_grid.
This uses bank U_grid, not the continuous bisection oracle used in final realized operational regret.
This is a diagnostic, not a final performance table or physical-safety certification.

## Horizon 3

- Myopic adaptation over calibrated Fixed: 0.000260; paired 95% interval [-0.003802, 0.004010].
- Combined lookahead gain over calibrated Fixed: 0.001823; paired 95% interval [-0.000573, 0.004896].
- Non-myopic gain over Myopic: 0.001563; paired 95% interval [-0.000833, 0.004740].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 1.000.
- Lookahead median posterior ESS: 95.60.

## Limits

A weak approximate planner cannot prove there is no adaptive or non-myopic space.
Screen and confirm use disjoint validation halves. Freeze settings before confirmation; final held-out tests remain separate.
A positive planning signal does not guarantee DAD can learn it. More distinct action sequences alone do not establish valuable adaptivity.
Fixed is approximate; a stronger Fixed calibration may reduce the apparent adaptive advantage.
Intervals are unadjusted per comparison and conditional on the banks, planning seeds and Fixed calibration.
Confirmation halves are consumed after inspection; further tuning requires fresh validation for confirmatory claims.
The prior Bayes risk is an in-model perfect-information ceiling, not a bound on off-support empirical gains.
