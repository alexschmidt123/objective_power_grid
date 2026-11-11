# Frozen HEC-v2 internal ablation

|Configuration|EIG nats|MCSE|
|---|---:|---:|
|full|12.929353|0.035016|
|expert1|11.193217|0.050998|
|expert2|11.942015|0.047489|
|expert3|11.421067|0.049217|
|expert4|12.427823|0.041914|
|uniform|12.937944|0.033807|
|conventional_moe|12.120527|0.044393|

2000 full trajectories,T30,1M negative contrasts plus positive once. Finite-contrast lower bound,not exact EIG. One training lineage;the original independent-from-training development bank is reused for this diagnostic. No final-test or retraining.

Best observed individual expert:expert4
Full policy reproduces its saved scores with maximum error:0.0
