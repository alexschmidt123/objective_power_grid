# Stronger IEEE9 duration audit

2 duration sets screened; 4 fresh validation systems.
Primary metric: finite-loss realized regret against the continuous-control oracle.
Positive gains favor the named adaptive policy. Adjusted intervals include all finalist/horizon/contrast comparisons.
Approximate Fixed and rolling two-step planning remain limitations; validation assumes the specified physical/prior/noise model.

## [0.2, 0.76, 1.32, 1.88, 2.44, 3.0]; T=2

- myopic_adaptive_gain: -0.002500; adjusted interval [-0.007500000000000007, 0.0].
- adaptive_gain: -0.007500; adjusted interval [-0.017500000000000016, 0.0].
- nonmyopic_gain: -0.005000; adjusted interval [-0.010000000000000009, 0.0].

## [0.28, 0.32, 0.37, 0.4, 0.47, 0.62]; T=2

- myopic_adaptive_gain: -0.002500; adjusted interval [-0.007500000000000007, 0.0].
- adaptive_gain: -0.000000; adjusted interval [-0.010000000000000009, 0.009999999999999995].
- nonmyopic_gain: 0.002500; adjusted interval [-0.010000000000000009, 0.0175].

