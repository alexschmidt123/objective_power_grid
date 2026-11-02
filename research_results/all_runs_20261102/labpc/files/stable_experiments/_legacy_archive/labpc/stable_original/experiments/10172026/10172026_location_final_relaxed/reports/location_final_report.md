# Location-finding final report

Frozen original relaxed routing; no warmup or straight-through replacements. Six baselines and exactly four MoE variants. Each trained model has1M total primary episodes; Random is untrained. All models are final-budget checkpoints. Fresh final evaluation is independent of development1001; the existing training seeds have informed development, so this is not a fresh independently designed training replication. No post-final tuning.

| Method | Final EIG mean | Seed SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Random | 8.20343 | N/A | 8.20343 | 8.20343 |
| Optimized Fixed | 8.71796 | 0.03460608918824462 | 8.66627 | 8.74855 |
| DAD | 9.89686 | 0.15537175947242707 | 9.72977 | 10.07055 |
| Simplified RL-sBOED | 9.40792 | 0.06035358207000891 | 9.31142 | 9.47514 |
| vsOED | 8.44579 | 0.29896178140011015 | 7.94912 | 8.67697 |
| ScoreBED | 9.27005 | 1.2435792273061632 | 7.10396 | 10.20361 |
| MoE-sBOED | 9.36844 | 1.780558578943339 | 6.28506 | 10.64245 |
| ResMoE-sBOED | 10.54055 | 0.1297101230284169 | 10.33505 | 10.68440 |
| SwitchMoE-sBOED | 9.57540 | 1.7215432059596434 | 6.56432 | 10.76241 |
| SwitchResMoE-sBOED | 10.53124 | 0.1038724245329497 | 10.38346 | 10.66349 |

## Predeclared paired comparisons

Evidence labels use simultaneous95% paired training-seed intervals across six contrasts, not conditional trajectory MC intervals.

- resmoe minus scorebed: +1.27050, simultaneous interval [-1.20301, 3.74400]: inconclusive.
- resmoe minus dad: +0.64368, simultaneous interval [0.18312, 1.10425]: positive_supported.
- resmoe minus moe: +1.17211, simultaneous interval [-2.71075, 5.05496]: inconclusive.
- switchmoe minus moe: +0.20696, simultaneous interval [-0.01594, 0.42987]: inconclusive.
- switchresmoe minus resmoe: -0.00931, simultaneous interval [-0.08528, 0.06667]: inconclusive.
- switchresmoe minus switchmoe: +0.95584, simultaneous interval [-2.81300, 4.72467]: inconclusive.

## Mechanism and publication limits

Saved diagnostics report unused experts, gradients and rerouting behavior. Existence of a gate or residual experts does not demonstrate useful persistence or specialization. ScoreBED crossings are diagnostic only. The single-residual forced-mode comparison is an intervention with distribution-shift limitations. Native baselines have disclosed benchmark/budget adaptations; simplified RL-sBOED is not an exact paper reproduction. Equal episodes do not imply equal runtime/FLOPs. Missing native likelihood-work counters and standalone latency measurements are explicitly unmeasured. Five seed runs give limited precision for small effects. This artifact supports a transparent fixed-budget study and does not guarantee journal acceptance or universal MoE superiority.
