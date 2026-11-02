## Official EIG and verified paired comparisons

Official seed1001 EIG: **10.81546708 +/- 0.04607956 nats** (trajectory MC SE conditional on the shared bank),2000 trajectories,1000000 contrasts,T30.

| Reference | Reference mean | IR minus reference | Paired MC SE | Approximate95% CI | N |
|---|---:|---:|---:|---|---:|
| DeepSeekMoE-v1 | 10.98906590 | -0.17359881 | 0.05403101 | [-0.27949959, -0.06769804] | 2000 |
| vsOED | 12.48084689 | -1.66537980 | 0.05684466 | [-1.77679533, -1.55396428] | 2000 |
| Corrected DAD | 10.27611286 | 0.53935422 | 0.06107940 | [0.41963860, 0.65906985] | 2000 |

Truth samples, standard-normal observation noise, trajectory order and contrast-bank bytes were reverified for vsOED and corrected DAD. The DeepSeekMoE-v1 record has the same saved draw identity. Paired SEs come from paired trajectory-score differences, not method-level means/SEs. The legacy DAD result is excluded. The one training seed is not multi-seed evidence.

Previously reserved confirmation seed1079 gave 10.90325091 +/- 0.04336984 nats. Neither official nor confirmation results were used to retune or replace the checkpoint.
