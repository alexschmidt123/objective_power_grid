# MoE-sBOED encoder correction study

Status: encoder_trial_has_not_demonstrated_a_DAD_win

History encoder is the largest controlled EIG bottleneck in seed101; routing/clipping contributes but hard/relaxed inference mismatch is negligible.

| Control | EIG | Total parameters | Active parameters |
|---|---:|---:|---:|
| A | 10.66260 | 58900 | 58898 |
| B | 8.31631 | 58564 | 58562 |
| C | 8.24200 | 84302 | 84300 |
| D | 8.65004 | 72208 | 22088 |
| capacity | 5.54353 | 84660 | 84658 |

Correction: Replace location attention history with original DAD sum16 set encoder, fixed padding and unchanged stage feature. Four expert heads and DSelect1 remain unchanged.

| Training seed | Original MoE | Corrected MoE | DAD |
|---|---:|---:|---:|
| 101 | 8.65004 | 9.38016 | 9.76665 |
| 102 | Pending | Pending | Pending |
| 103 | Pending | Pending | Pending |

| Model | EIG mean across completed training seeds | Sample SD | Seeds |
|---|---:|---:|---:|
| original | 8.650043989674257 | None | 1 |
| corrected | 9.380160554227764 | None | 1 |
| dad | 9.76665434381799 | None | 1 |

| Reused baseline, seed101 | EIG | MSE |
|---|---:|---:|
| random | 8.26624 | 0.07899 |
| fixed | 8.85782 | 0.08118 |
| dad | 9.76665 | 0.07231 |
| rl_sboed | 9.49698 | 0.05388 |
| vsoed | 6.62949 | 0.21104 |
| scorebed | 9.91568 | 0.08104 |

Policy failures are supported by paired information deficits and persistent errors under32768particles/32moves/3seeds. Posterior multimodality and budget sensitivity affect some point estimates; label-permutation error is already removed. Three selected cases cannot certify all2000 inferences.

Encoder correction improves the original MoE but remains below DAD. Three-seed runs were not launched because the DAD-win criterion failed. No sparse-MoE advantage has been demonstrated; remaining expert/head/routing effects need matched controls.

All formal runs retain1million training episodes, L1024 training and2000/L1000000evaluation. Timing and smoke results are correctness/cost diagnostics only.

Implementation files: src/research/sparse_moe.py, src/research/moe_architecture_controls.py, src/research/runner.py. Original attention remains the default.
