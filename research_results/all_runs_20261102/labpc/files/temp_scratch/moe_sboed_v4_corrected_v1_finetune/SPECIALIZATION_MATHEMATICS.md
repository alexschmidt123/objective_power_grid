# Offline gradient adaptation
Original loss L=-mean_i J_i(T)+B_router is unchanged. With detached cohort masks m_ic, define L_c=-mean_i m_ic J_i(T)+mean_i(m_ic)B_router. Three masks (reliable low, reliable high, unreliable) partition rows, so sum_c grad L_c=grad L.
For each routed expert use its existing selected-expert softmax exposure to estimate affinity in the two reliable cohorts. The historical protect_gradient projects only conflicting cohort components and guards original descent alignment. The proposed corrected direction adds the unreliable-cohort contribution unchanged, requires positive inner products against both original full-loss and historical-adapted gradients, and matches historical expert-gradient norm. Fallback is the exact historical contextual-gradient result. Encoder/shared/router/emitter gradients are never changed by this procedure.
This is a guarded optimizer heuristic, not an unbiased EIG gradient or new information decomposition. No role auxiliary loss is introduced. Actual application requires validated cohort coverage; that gate failed.


Status: BLOCKED_INFORMATION_COHORT_VALIDATION. No fine-tuning or official evaluation was launched. No STABLE changes, GitHub push or changes to previous V4/Ubuntu SIR.
