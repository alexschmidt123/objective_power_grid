# Corrected IEEE9 duration-combination audit

Screened 2 of 647,992,090,956 six-duration combinations.
Finite-loss 95% quantile; candidate-grid bank regret; exploratory reused validation.
Positive gains favor lookahead. Intervals use theta-cluster bootstrap with Bonferroni tail adjustment across all reported confirmation contrasts.

## Durations [0.2, 0.76, 1.32, 1.88, 2.44, 3.0]; T=2

- myopic_adaptive_gain: -0.004375; adjusted interval [-0.018750000000000013, 0.009167317708333383].
- adaptive_gain: -0.013281; adjusted interval [-0.031406250000000024, -0.0007812499999999958].
- nonmyopic_gain: -0.008906; adjusted interval [-0.030781250000000027, 0.011093750000000012].

## Durations [0.28, 0.32, 0.37, 0.4, 0.47, 0.62]; T=2

- myopic_adaptive_gain: 0.012188; adjusted interval [-0.00015625000000000418, 0.029323567708333394].
- adaptive_gain: 0.009063; adjusted interval [-0.00125, 0.02484375000000002].
- nonmyopic_gain: -0.003125; adjusted interval [-0.018281250000000013, 0.010312500000000011].

Approximate Fixed and rolling two-step planner; no no-room theorem.
Validation reused during earlier development; exploratory, not fresh confirmation.
Final test observations excluded. Continuous-oracle validation still required.
