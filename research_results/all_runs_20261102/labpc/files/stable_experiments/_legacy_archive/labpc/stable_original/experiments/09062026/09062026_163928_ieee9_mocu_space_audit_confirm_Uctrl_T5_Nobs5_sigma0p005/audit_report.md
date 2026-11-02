# Finite-loss MOCU space audit

Phase: confirm. 64 validation systems × 3 observation-noise seeds; no policy training.
Posterior support: 384; actions: 54; prior Bayes risk: 0.028958.

Positive gain favors lookahead. Confidence intervals cluster repeated seeds by validation system.
This is a diagnostic, not a final performance table or physical-safety certification.

## Horizon 2

- Adaptive gain over calibrated Fixed: 0.001979; paired 95% interval [-0.002031, 0.006250].
- Non-myopic gain over Myopic: -0.001458; paired 95% interval [-0.005833, 0.002760].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.984.

## Horizon 3

- Adaptive gain over calibrated Fixed: -0.001719; paired 95% interval [-0.006302, 0.003021].
- Non-myopic gain over Myopic: 0.001615; paired 95% interval [-0.002553, 0.005833].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.979.

## Horizon 4

- Adaptive gain over calibrated Fixed: -0.001042; paired 95% interval [-0.006876, 0.004740].
- Non-myopic gain over Myopic: 0.000990; paired 95% interval [-0.005887, 0.008438].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.964.

## Horizon 5

- Adaptive gain over calibrated Fixed: 0.002813; paired 95% interval [-0.003543, 0.008803].
- Non-myopic gain over Myopic: 0.002031; paired 95% interval [-0.002240, 0.006355].
- Status: `inconclusive_increase_planning_or_validation_budget`.
- Lookahead bank-based coverage: 0.979.

## Limits

A weak approximate planner cannot prove there is no adaptive or non-myopic space.
Screen and confirm use disjoint validation halves. Freeze settings before confirmation; final held-out tests remain separate.
A positive planning signal does not guarantee DAD can learn it. More distinct action sequences alone do not establish valuable adaptivity.
Fixed is approximate; a stronger Fixed calibration may reduce the apparent adaptive advantage.
The prior Bayes risk is an in-model perfect-information ceiling, not a bound on off-support empirical gains.
