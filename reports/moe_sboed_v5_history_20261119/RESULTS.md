# Matched MoE-sBOED V5 seed101 pilot

Status: COMPLETE.

|Method|EIG lower bound (nats)|MCSE|
|---|---:|---:|
|MoE-sBOED V5|12.88075|0.03560|
|MoE-sBOED V5-uniform|12.77441|0.03871|
|Original V4 learned|12.77001|0.03825|
|Original V4-uniform|12.71567|0.03850|
|vsOED|12.45772|0.04110|

|Paired comparison|Difference|95% evaluation CI|
|---|---:|---|
|V5_minus_V4|+0.11074|[+0.05835,+0.16314]|
|learned_minus_uniform|+0.10634|[+0.05002,+0.16266]|
|V5_minus_vsOED|+0.42303|[+0.32562,+0.52045]|
|uniform_V5_minus_V4|+0.05874|[-0.00423,+0.12172]|
|original_routing_gap|+0.05434|[-0.00343,+0.11210]|
|gap_improvement|+0.05200|[-0.02164,+0.12565]|

Decision: {"primary_credible": true, "secondary_credible": true, "uniform_change": {"n": 2000, "mean": 0.05874172151697762, "MCSE": 0.03211160587343243, "CI95": [-0.004233999945661739, 0.12171744297961698]}, "automatic_expansion": false, "recommend_five_seeds": true, "scope": "One continuation seed,case MC uncertainty;not seed robustness. History encoder and expert conditioning are one joint intervention;not separately attributed."}. No automatic expansion.

Finite-contrast EIG lower bounds;MCSE does not include contrast bias. Shared inherited pretraining;one continuation seed.
