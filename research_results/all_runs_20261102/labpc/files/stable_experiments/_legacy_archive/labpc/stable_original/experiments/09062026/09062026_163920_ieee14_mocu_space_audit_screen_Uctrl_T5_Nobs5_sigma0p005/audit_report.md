# Finite-loss MOCU space audit

Phase: screen. 64 validation systems × 3 observation-noise seeds; no policy training.
Posterior support: 384; actions: 84; prior Bayes risk: 0.023307.

Positive gain favors lookahead. Confidence intervals cluster repeated seeds by validation system.
This is a diagnostic, not a final performance table or physical-safety certification.

## Horizon 2

- Adaptive gain over calibrated Fixed: 0.000625; paired 95% interval [-0.000365, 0.001563].
- Non-myopic gain over Myopic: 0.000156; paired 95% interval [-0.000625, 0.000990].
- Status: `material_gain_not_shown_by_this_planner`.
- Lookahead bank-based coverage: 1.000.

## Horizon 3

- Adaptive gain over calibrated Fixed: 0.001667; paired 95% interval [-0.001771, 0.005000].
- Non-myopic gain over Myopic: 0.001406; paired 95% interval [-0.001928, 0.005000].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.995.

## Horizon 4

- Adaptive gain over calibrated Fixed: 0.002344; paired 95% interval [-0.000937, 0.007605].
- Non-myopic gain over Myopic: 0.001042; paired 95% interval [-0.000625, 0.003490].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 1.000.

## Horizon 5

- Adaptive gain over calibrated Fixed: 0.000417; paired 95% interval [-0.003646, 0.004324].
- Non-myopic gain over Myopic: 0.001510; paired 95% interval [-0.003385, 0.007501].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.990.

## Limits

A weak approximate planner cannot prove there is no adaptive or non-myopic space.
Screen and confirm use disjoint validation halves. Freeze settings before confirmation; final held-out tests remain separate.
A positive planning signal does not guarantee DAD can learn it. More distinct action sequences alone do not establish valuable adaptivity.
Fixed is approximate; a stronger Fixed calibration may reduce the apparent adaptive advantage.
The prior Bayes risk is an in-model perfect-information ceiling, not a bound on off-support empirical gains.
