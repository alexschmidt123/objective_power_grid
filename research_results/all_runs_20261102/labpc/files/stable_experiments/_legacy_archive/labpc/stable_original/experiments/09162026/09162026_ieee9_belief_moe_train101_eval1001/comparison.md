# IEEE9 signed endpoint RoCoF, 3.5 seconds

One training seed (101), one evaluation seed (1001), 128 test systems, 128 contrasts.
Mean sPCE in nats; no across-seed SD is available. Training algorithms and compute costs differ.
Belief MoE uses 32,000 behavior episodes plus 64,000 posterior-predictive continuations per horizon. Baseline training budgets are preserved in their source records. This is not a matched-compute comparison.

| Method | T=3 | T=4 | T=5 |
|---|---:|---:|---:|
| moe_sboed | 2.1163 | 2.3217 | 2.5336 |
| dad | 2.1742 | 2.3064 | 2.6181 |
| rl_sboed | 2.1706 | 2.3407 | 2.5361 |
| step_dad | 2.2245 | 2.3644 | 2.6154 |
| myopic | 1.8824 | 2.0714 | 2.2365 |
| fixed | 2.0124 | 2.2651 | 2.5181 |
| random | 0.4636 | 0.5130 | 0.5072 |
