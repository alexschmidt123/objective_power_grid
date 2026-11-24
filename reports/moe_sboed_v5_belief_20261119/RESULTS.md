# New V5:offline native-belief refresh,seed101

Status:COMPLETE. ORIGINAL canonical V2/V4/DAD/vsOED bank,no retraining controls.

|Method|EIG lower bound|MCSE|
|---|---:|---:|
|MoE-sBOED V5|12.83194|0.03581|
|MoE-sBOED V5-uniform|12.78853|0.03541|
|Original V4 learned|12.85611|0.03599|
|Original V4-uniform|12.80587|0.03588|
|vsOED|12.44272|0.04209|

|Paired comparison|Difference|95% case evaluation CI|
|---|---:|---|
|newV5_minus_uniform|+0.04341|[-0.01200,+0.09882]|
|newV5_minus_previous_learned|-0.02417|[-0.08316,+0.03482]|
|newV5_minus_previous_uniform|+0.02607|[-0.03216,+0.08429]|
|newV5_minus_vsoed|+0.38922|[+0.29325,+0.48519]|
|uniform_minus_V4uniform|-0.01734|[-0.07695,+0.04227]|

One continuation seed,not training-seed robustness. Finite1M-contrast lower bounds;MCSE excludes contrast bias. No automatic expansion or canonical-bank tuning.
