# Finite-loss MOCU space audit

Phase: confirm. 64 validation systems × 3 observation-noise seeds; no policy training.
Posterior support: 384; actions: 84; prior Bayes risk: 0.023307.

Positive gain favors lookahead. Confidence intervals cluster repeated seeds by validation system.
This is a diagnostic, not a final performance table or physical-safety certification.

## Horizon 2

- Adaptive gain over calibrated Fixed: 0.000573; paired 95% interval [-0.001146, 0.003073].
- Non-myopic gain over Myopic: -0.000365; paired 95% interval [-0.000937, 0.000208].
- Status: `material_gain_not_shown_by_this_planner`.
- Lookahead bank-based coverage: 1.000.

## Horizon 3

- Adaptive gain over calibrated Fixed: 0.001615; paired 95% interval [-0.000625, 0.004845].
- Non-myopic gain over Myopic: -0.000625; paired 95% interval [-0.004688, 0.003490].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.990.

## Horizon 4

- Adaptive gain over calibrated Fixed: 0.000156; paired 95% interval [-0.003646, 0.004063].
- Non-myopic gain over Myopic: -0.001146; paired 95% interval [-0.004688, 0.001979].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.990.

## Horizon 5

- Adaptive gain over calibrated Fixed: -0.001198; paired 95% interval [-0.006302, 0.003802].
- Non-myopic gain over Myopic: -0.003490; paired 95% interval [-0.007760, -0.000104].
- Status: `material_gain_not_shown_by_this_planner`.
- Lookahead bank-based coverage: 0.979.

## Limits

A weak approximate planner cannot prove there is no adaptive or non-myopic space.
Screen and confirm use disjoint validation halves. Freeze settings before confirmation; final held-out tests remain separate.
A positive planning signal does not guarantee DAD can learn it. More distinct action sequences alone do not establish valuable adaptivity.
Fixed is approximate; a stronger Fixed calibration may reduce the apparent adaptive advantage.
The prior Bayes risk is an in-model perfect-information ceiling, not a bound on off-support empirical gains.
