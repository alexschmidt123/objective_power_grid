# Online time (seconds per rollout)

| Method | T=3 | T=4 | T=5 |
| --- | ---: | ---: | ---: |
| DAD | 0.005209 ± 0.000068 | — | 0.040363 ± 0.010478 |
| RL-SBOED | 0.004661 ± 0.000057 | — | 0.020089 ± 0.002371 |
| Step-DAD | 6.223860 ± 0.059371 | — | 23.931939 ± 1.666104 |
| Myopic | 0.006268 ± 0.000011 | — | 0.022011 ± 0.001623 |
| Fixed | **0.000381 ± 0.000002** | — | **0.001964 ± 0.000180** |
| Random | 0.000433 ± 0.000002 | — | 0.002187 ± 0.000189 |

Trained methods: 3 training seeds × 5 evaluation seeds. Baselines: 5 unique evaluation seeds. Offline time: 3 run-level values.
