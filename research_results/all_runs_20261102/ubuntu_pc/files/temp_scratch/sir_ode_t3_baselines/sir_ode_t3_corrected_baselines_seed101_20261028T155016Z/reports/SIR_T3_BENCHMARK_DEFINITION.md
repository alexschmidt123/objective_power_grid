# SIR T3 continuous benchmark v1

Independent development campaign; no T5 checkpoint or historical source was overwritten. S0=498,I0=2,R0=0,N500. dS=-beta*S*I/N,dI=beta*S*I/N-gamma*I,dR=gamma*I. Independent log rates Normal(log(.5),.5²), Normal(log(.1),.5²). Observations I(t)+Normal(0,1), exactly three chronological measurements through100 with minimum gap.1. Physical state is propagated between measurements; no reset, no true theta in policies.

Preserved T5 chronological map: lower=previous_time+.1 (initial previous0), upper=100-(3-stage-1)*.1; sigmoid/tanh-affine unit action maps into this interval. Consequently the inherited implementation uses t1>=.1, a subset of the nominal0<=t1 domain; this is explicit, common across methods, and was not changed silently. Random is uniform CONDITIONAL on each remaining feasible interval, not uniform over the joint ordered-simplex volume.

Numerical solver: float64 rate-adaptive RK4, minimum512 substeps per interval; counts=max(512,ceil(duration*(beta+gamma)/.2)), detached for derivatives. The new T3-only kernel removes zero-length padding beyond each CUDA block maximum and enables validated float64 multiply-add fusion. No interpolation, solver tolerance relaxation, prior/noise alteration, discrete action bank, or state reset.
