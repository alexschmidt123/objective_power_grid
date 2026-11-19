# Replacement V5: PRIMARY canonical evaluation,seed101

Original V2/V4/DAD/vsOED bank;2000 trajectories,1M contrasts,T30. Supplementary development-bank results remain unchanged.

|Method|EIG lower bound (nats)|MCSE|
|---|---:|---:|
|MoE-sBOED V5|12.89320|0.03467|
|MoE-sBOED V5-uniform|12.86427|0.03544|
|Original V4 learned|12.85611|0.03599|
|Original V4-uniform|12.80587|0.03588|
|vsOED|12.44272|0.04209|
|DAD|10.31407|0.04848|

|Paired comparison|Difference|95% case evaluation CI|
|---|---:|---|
|V5_minus_V4|+0.03709|[-0.01778,+0.09196]|
|learned_minus_uniform|+0.02892|[-0.02861,+0.08646]|
|V5_minus_vsOED|+0.45048|[+0.35452,+0.54644]|
|uniform_V5_minus_V4|+0.05840|[-0.00270,+0.11951]|
|V5_minus_DAD|+2.57912|[+2.47319,+2.68506]|
|gap_improvement|-0.02132|[-0.09991,+0.05728]|

Fresh V4 learned/uniform and vsOED score vectors reproduce saved canonical controls within1e-9. DAD canonical scores reused after saved-score/checkpoint hash validation. Historical V2/DAD/vsOED guard certificate and bank/scorer hashes validated. No training/checkpoint selection. One continuation seed,not five-seed robustness.
