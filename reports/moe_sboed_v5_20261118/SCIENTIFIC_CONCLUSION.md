# MoE-sBOED V5 scientific conclusion

- V5_minus_V4: -0.04236 nats;95% paired evaluation CI [-0.10071,+0.01600].
- learned_minus_uniform: -0.02903 nats;95% paired evaluation CI [-0.08179,+0.02374].
- V5_minus_vsOED: +0.41844 nats;95% paired evaluation CI [+0.32267,+0.51420].
- uniform_V5_minus_V4: +0.07831 nats;95% paired evaluation CI [+0.02027,+0.13636].
- original_routing_gap: +0.09164 nats;95% paired evaluation CI [+0.03404,+0.14925].
- gap_improvement: -0.12067 nats;95% paired evaluation CI [-0.20343,-0.03791].

The primary absolute-EIG improvement was not established. Stop this architectural pilot;no expansion or additional tuning.

Recommendation: Do not expand to five seeds on this pilot evidence.

Controller state: {"status": "COMPLETE", "completed": ["learned", "uniform", "previous_learned", "previous_uniform", "vsoed", "sensitivity"], "automatic_expansion": false}.

All primary estimates are finite1M-contrast EIG lower bounds. Known estimator saturation/contrast bias is not represented by the MCSE. Optional sensitivity status: NOT_TRIGGERED. This is one continuation seed from shared V2 pretraining,not seed robustness or independent from-scratch training.

V5 changes both weighted expert-slot preservation and head capacity. An improvement cannot identify their individual causal effects. Learned routing must be assessed through paired EIG,not weight variation or gradient norms. No canonical-bank checkpoint/hyperparameter selection,no teacher,no new rewards,no V4-Spec masking,and no automatic extra seeds.
