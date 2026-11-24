SUPPLEMENTARY development-bank analysis. Primary canonical conclusions supersede any expansion recommendation below;see reports/moe_sboed_v5_history_canonical_20261119/SCIENTIFIC_CONCLUSION.md.

# MoE-sBOED V5 scientific conclusion

- V5_minus_V4: +0.11074 nats;95% paired evaluation CI [+0.05835,+0.16314].
- learned_minus_uniform: +0.10634 nats;95% paired evaluation CI [+0.05002,+0.16266].
- V5_minus_vsOED: +0.42303 nats;95% paired evaluation CI [+0.32562,+0.52045].
- uniform_V5_minus_V4: +0.05874 nats;95% paired evaluation CI [-0.00423,+0.12172].
- original_routing_gap: +0.05434 nats;95% paired evaluation CI [-0.00343,+0.11210].
- gap_improvement: +0.05200 nats;95% paired evaluation CI [-0.02164,+0.12565].

Primary absolute-EIG improvement has a positive paired interval. Additional seeds may be recommended subject to uniform performance and separate authorization.

Recommendation: Consider separately authorized continuation seeds102-105.

Controller state: {"status": "COMPLETE", "completed": ["learned", "uniform", "previous_learned", "previous_uniform", "vsoed", "sensitivity"], "automatic_expansion": false}.

All primary estimates are finite1M-contrast EIG lower bounds. Known estimator saturation/contrast bias is not represented by the MCSE. Optional sensitivity status: COMPLETE. This is one continuation seed from shared V2 pretraining,not seed robustness or independent from-scratch training.

Replacement V5 changes the ordered history encoder and expert conditioning jointly;their individual effects are not identified. V4 fusion and action-head capacity are preserved. Learned routing must be assessed through paired EIG,not weight variation or gradient norms. No canonical-bank checkpoint/hyperparameter selection,no teacher,no new rewards,no V4-Spec masking,and no automatic extra seeds.
