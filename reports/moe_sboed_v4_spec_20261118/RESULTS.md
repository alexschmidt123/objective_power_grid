# V4-Spec matched seed101 pilot

Status: COMPLETE. Same NEW bank for all five policies;original canonical/pilot results unchanged.

|Method|EIG nats|MCSE|
|---|---:|---:|
|MoE-sBOED-V4-Spec|12.83268|0.03546|
|MoE-sBOED-V4-Spec-uniform|12.79506|0.03598|
|Original V4 learned|12.84314|0.03677|
|Original V4-uniform|12.80603|0.03657|
|vsOED|12.36209|0.04323|

|Paired comparison|Difference|95% evaluation CI|
|---|---:|---|
|learned_minus_uniform|+0.03762|[-0.01293,+0.08817]|
|Spec_minus_original|-0.01046|[-0.06743,+0.04650]|
|Spec_minus_vsOED|+0.47059|[+0.37206,+0.56912]|
|original_routing_gap|+0.03711|[-0.02206,+0.09629]|
|gap_improvement|+0.00051|[-0.07796,+0.07897]|
|uniform_change|-0.01097|[-0.07120,+0.04926]|

Decision: {"improvement_A": false, "improvement_B": false, "uniform_not_deteriorated_point_estimate": false, "recommend_replication": false, "automatic_expansion": false, "scope": "One-seed development pilot,case uncertainty conditional on bank;not training-seed robustness or proof of optimal expert regimes"}. No automatic expansion.
