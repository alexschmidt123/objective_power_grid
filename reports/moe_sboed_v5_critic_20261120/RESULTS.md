# New V5:training-only belief-aware native critic,seed101

Status:COMPLETE. Original canonical bank,2000 cases,1M contrasts,T30.

|Method|EIG lower bound|MCSE|
|---|---:|---:|
|MoE-sBOED V5|13.00472|0.03375|
|MoE-sBOED V5-uniform|12.96689|0.03337|
|Previous V5 learned|12.89320|0.03467|
|Previous V5 uniform|12.86427|0.03544|
|Original V4 learned|12.85611|0.03599|
|vsOED|12.44272|0.04209|

|Paired comparison|Difference|95% case CI|
|---|---:|---|
|newV5_minus_uniform|+0.03783|[-0.01601,+0.09166]|
|newV5_minus_reference_learned|+0.11152|[+0.05321,+0.16983]|
|newV5_minus_reference_previous_learned|+0.14861|[+0.08709,+0.21013]|
|newV5_minus_reference_vsoed|+0.56200|[+0.46735,+0.65664]|
|uniform_minus_previous_uniform|+0.10262|[+0.04068,+0.16455]|

One continuation seed,not seed robustness. Finite-contrast lower bounds;MCSE excludes contrast bias. No automatic expansion.
