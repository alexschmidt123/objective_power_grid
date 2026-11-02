# IR-MoE-v2 seed101 results

Official historical benchmark: 10.77158549 +/- 0.04424949 nats (trajectory MC SE conditional on the shared bank), 2000 cases/L1000000/T30.

Reserved seed93751001: 10.78051213 +/- 0.04441015 nats.

| Reused reference | Mean EIG | V2 minus reference (descriptive unless verified paired) |
|---|---:|---:|
| vsOED | 12.48085 | -1.70926 |
| PIMoE-IC | 11.88809 | -1.11650 |
| Original_PIMoE | 11.41594 | -0.64435 |
| MoE-v2 | 11.04023 | -0.26864 |
| DeepSeekMoE-v1 | 10.98907 | -0.21748 |
| Corrected_DAD | 10.27611 | 0.49548 |
| Random | 8.26624 | 2.50535 |

Learned-minus-uniform development difference: -0.02776086 +/- 0.01559881 nats, paired MC SE. Online inference mean 0.70969 ms, median 0.70998 ms.

One training seed. No retuning followed official evaluation. No external host was queried for checkpoint data. Matching original IR and DeepSeekMoE-v1 checkpoints were unavailable locally. See paired_comparisons.json for verified comparisons.
