# SIR ODE MoE ablations

Mean ± sample SD of finite-particle EIG across evaluation seeds 1001–1005 (nats). Training seed 101; 512 systems per evaluation. SD is evaluation noise variability, not training variability.

| Policy | T=3 | T=4 | T=5 |
|---|---:|---:|---:|
| MoE-sBOED | 6.8525 ± 0.0030 | 6.8702 ± 0.0046 | 6.8817 ± 0.0041 |
| Matched single network | 6.8516 ± 0.0030 | 6.8699 ± 0.0047 | 6.8849 ± 0.0061 |
| Uniform mixture (trained) | 6.8535 ± 0.0064 | 6.8706 ± 0.0038 | 6.8795 ± 0.0058 |
| Uniform weights (frozen MoE) | 6.4113 ± 0.0093 | 6.3701 ± 0.0061 | 6.7059 ± 0.0052 |
| Expert 1 alone | 6.1207 ± 0.0145 | 6.1603 ± 0.0043 | 6.6641 ± 0.0044 |
| Expert 2 alone | 5.6386 ± 0.0226 | 5.4600 ± 0.0110 | 6.4986 ± 0.0075 |
| Expert 3 alone | 6.7369 ± 0.0077 | 6.8465 ± 0.0050 | 6.8390 ± 0.0048 |
| Expert 4 alone | 5.2645 ± 0.0134 | 5.4083 ± 0.0128 | 6.5454 ± 0.0102 |
| Single expert selected on validation | 6.7369 ± 0.0077 | 6.8465 ± 0.0050 | 6.8390 ± 0.0048 |

Matched single network and uniform mixture are freshly trained for 200 epochs × 512 trajectories. Frozen interventions use the original MoE checkpoint without updating any parameters.

The validation-selected expert is chosen separately for each horizon before inspecting test scores. Expert identities are internal to each separately trained horizon model.

One training seed; five evaluation seeds do not replace independent training repetitions.
Single-head and uniform interventions use co-trained frozen MoE heads and can visit out-of-routing-distribution states.
Parameter matching does not match activated FLOPs.
Entropy-reduction estimator is bounded above by log(1024) = 6.93147 nats.
