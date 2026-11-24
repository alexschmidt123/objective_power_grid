# Matched MoE-sBOED V5 seed101 pilot

Status: COMPLETE.

|Method|EIG lower bound (nats)|MCSE|
|---|---:|---:|
|MoE-sBOED V5|12.83576|0.03721|
|MoE-sBOED V5-uniform|12.86479|0.03594|
|Original V4 learned|12.87812|0.03688|
|Original V4-uniform|12.78648|0.03710|
|vsOED|12.41733|0.04209|

|Paired comparison|Difference|95% evaluation CI|
|---|---:|---|
|V5_minus_V4|-0.04236|[-0.10071,+0.01600]|
|learned_minus_uniform|-0.02903|[-0.08179,+0.02374]|
|V5_minus_vsOED|+0.41844|[+0.32267,+0.51420]|
|uniform_V5_minus_V4|+0.07831|[+0.02027,+0.13636]|
|original_routing_gap|+0.09164|[+0.03404,+0.14925]|
|gap_improvement|-0.12067|[-0.20343,-0.03791]|

Decision: {"primary_credible": false, "secondary_credible": false, "uniform_change": {"n": 2000, "mean": 0.07831434384447035, "MCSE": 0.029595604257039325, "CI95": [0.020272882524971766, 0.13635580516396895]}, "automatic_expansion": false, "recommend_five_seeds": false, "scope": "One continuation seed,case MC uncertainty;not seed robustness. Slot preservation and head capacity confounded."}. No automatic expansion.

Finite-contrast EIG lower bounds;MCSE does not include contrast bias. Shared inherited pretraining;one continuation seed.
