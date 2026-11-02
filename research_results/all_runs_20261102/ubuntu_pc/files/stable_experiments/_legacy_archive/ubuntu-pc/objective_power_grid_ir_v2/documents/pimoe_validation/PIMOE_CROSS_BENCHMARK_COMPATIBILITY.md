# Cross-benchmark compatibility

Location finding:implemented and tested; T30, actiondim2, scalar Gaussian observations, paramdim4.

SIR ODE:existing benchmark and on-the-fly simulator are present, but current PIMoE is NOT directly compatible. `AdaptiveBelief._move` hardcodes4D covariance and [2,3,0,1]source swaps; prior target assumes standard-normal latent coordinates; responses ignore carried physical state and expect scalar observations; action features index bothx/y; anchors use a2D Cartesian grid; stage divides30. SIR uses chronological1Dtime designs, different prior coordinates and ODE state. The additive homoscedastic Gaussian entropy identity can transfer only after validating its observation model; the current conditional_entropy guard rejects SIR. Genuine benchmark adapters and inference tests are needed; no new SIR training launched.

IEEE9/IEEE14:existing swing/RoCoF benchmark backends exist, but PIMoE currently cannot transfer unchanged. Dimensions, priors, carried non-reset dynamics, probe-duration feasibility and possibly multi-output responses violate current assumptions. A generic response-space interface is conceptually possible but is NOT implemented. No grid training or adapter redesign is part of this validation.

Conclusion:the predictive-mixture idea is not intrinsically tied to source coordinates, but the ACTUAL prototype is location-specific. Do not claim cross-benchmark generality until adapters and posterior/entropy validation pass. This study does not modify the frozen algorithm to manufacture portability.
