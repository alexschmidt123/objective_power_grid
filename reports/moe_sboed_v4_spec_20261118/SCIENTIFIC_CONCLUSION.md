# Scientific conclusion: V4-Spec seed101

The prespecified same-bank paired comparisons are complete.

- learned_minus_uniform: +0.03762 nats;95% paired evaluation CI [-0.01293,+0.08817].
- Spec_minus_original: -0.01046 nats;95% paired evaluation CI [-0.06743,+0.04650].
- Spec_minus_vsOED: +0.47059 nats;95% paired evaluation CI [+0.37206,+0.56912].
- original_routing_gap: +0.03711 nats;95% paired evaluation CI [-0.02206,+0.09629].
- gap_improvement: +0.00051 nats;95% paired evaluation CI [-0.07796,+0.07897].
- uniform_change: -0.01097 nats;95% paired evaluation CI [-0.07120,+0.04926].

The complete improvement criteria were not met. Report this pilot as negative/inconclusive and stop;do not expand or tune.

This is one continuation seed with shared parent/provider training,not a robustness study. The new bank is independent of previous architecture-selection banks. Regimes are exposure heuristics,not optimal-expert labels. Nonuniform routing and feature differences alone do not demonstrate information value.

The canonical1M-contrast estimator has material saturation. A prespecified64-case diagnostic found substantial1M->4M shifts for original frozen policies;see SATURATION_SENSITIVITY.json. Primary estimates remain matched1M finite-contrast lower bounds,not exact EIG,and case MCSE does not cover finite-contrast bias. A positive pilot result is conditional on this estimator and needs independent estimator-resolution validation before a stronger scientific claim.
