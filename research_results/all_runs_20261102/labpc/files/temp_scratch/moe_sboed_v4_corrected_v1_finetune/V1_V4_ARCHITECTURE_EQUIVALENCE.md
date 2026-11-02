# Identical deployed architecture
Exact pinned DeepSeekPolicy code retained. strict=True loads successfully. 8434 total parameters;6898 activated per context;16-dimensional history features;1shared +7routed SwiGLU experts(width8);top3;original softmax selected gate weights (not renormalized), residual integration, emitter and feasible mapping.
All30 actions at identical observed inputs are bitwise equal before optimization. Both initial state_dicts are bitwise equal. No new inference module or posterior computation. Source comparisons to042915b are in foundation_source_verification.json.


Status: BLOCKED_INFORMATION_COHORT_VALIDATION. No fine-tuning or official evaluation was launched. No STABLE changes, GitHub push or changes to previous V4/Ubuntu SIR.
