# Independent MoE-sBOED SIR ODE test

Mean EIG (nats), train seed 101, five evaluation seeds, 512 systems per seed.

| Method | T=3 | T=4 | T=5 |
|---|---:|---:|---:|
| DAD | 6.8493 | 6.8634 | 6.8747 |
| RL-sBOED | 6.8523 | 6.8695 | 6.8831 |
| Step-DAD | 6.8513 | 6.8674 | 6.8880 |
| Myopic | 6.7108 | 6.7437 | 6.7596 |
| Fixed | 6.2816 | 6.4501 | 6.5360 |
| Random | 3.7783 | 3.8593 | 3.9241 |
| MoE-sBOED | 6.8525 | 6.8702 | 6.8817 |

MoE: four experts, learned top-2 sparse dispatch at every stage, fresh PPO training, no teacher or baseline checkpoint, validation-EIG checkpoint selection.

This is one training seed. Routing statistics and all source paths/checksums are in report.json; evaluation-seed variation is reported separately in summary.csv.
