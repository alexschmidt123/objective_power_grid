# On-the-fly SIR ODE study: benchmark clarification

This clarification records the already implemented simulator behavior. It does
not change the frozen training configurations or running campaign.

Every newly generated primary trajectory draws fresh beta and gamma from the
declared log-normal prior. Those rates remain fixed within the trajectory.
Starting at S=498, I=2, R=0, the deterministic SIR equations are integrated
sequentially at the observation times chosen by the policy:

`dS/dt = -beta*S*I/500`, `dI/dt = beta*S*I/500 - gamma*I`,
`dR/dt = gamma*I`.

Each observation is `I(t) + epsilon`, with independent Gaussian noise SD 1.
The horizon is T=5, times end at 100, and the feasible mapping enforces the
configured minimum spacing 0.1. No finite collection of simulated epidemic
paths, discretized parameter grid or trajectory lookup table supplies training
observations. CUDA fusion changes execution only, retaining adaptive discrete
RK4 and its first-order derivatives.

## Code audit

- `src/research/engine.py:rollout` samples new primary and contrast parameters
  from `benchmark.sample` on each rollout, carries each particle's own state,
  and generates Gaussian noise at every stage.
- `src/research/benchmarks.py:SIR` defines the log-normal prior, explicit
  Gaussian likelihood, carried SIR state, feasible times and ODE integration.
- `src/research/sir_upstream_vsoed.py:SIRNativeModel` generates fresh rate draws
  and observations through the same benchmark. Native replay reuses generated
  histories; it is not a precomputed simulator database.
- `src/research/sir_scorebed_model.py:SIRScoreModel` samples fresh parameters
  and noise in the native JAX trainer and integrates the matching ODE. Its
  separately counted normalization simulations do not form a simulator lookup.
- RL replay also reuses generated episodes under its native algorithm. Only
  newly simulated episodes count toward the one-million primary budget.

The fixed, common evaluation contrast bank is used for paired EIG scoring.
It is not a finite training simulator and does not constrain the training prior.
Validation uses independent fixed draws for diagnostics only.

## Scope of comparisons

Call this campaign the **on-the-fly SIR ODE study**. Keep its seed-101 development
results separate from historical campaigns. Some earlier repository ODE code
also supported on-the-fly integration; do not claim that every historical SIR
experiment used a data bank. In particular, the published stochastic/data-bank
SIR settings associated with native vsOED differ from this explicit-likelihood
ODE benchmark. Do not borrow their EIG values or present this as a reproduction
of that different physical/observation model.

All ten current methods must solve this same ODE benchmark. Report native
algorithm adaptations, development status, estimator resolution, training costs
and independent replication before making journal-level comparative claims.
