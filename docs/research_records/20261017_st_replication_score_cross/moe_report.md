# Straight-through MoE five-seed replication

Seeds103/105 are preserved prior runs;101/102/104 use identical frozen scientific source/configuration. Exactly1M primary episodes each; final-budget checkpoint.2000 paired trajectories/L1M/development1001. No original result replacement.

| Seed | Original EIG | ST hard EIG | ST relaxed EIG | Paired gain [95% MC interval] | E1/E2/E3/E4 usage |
|---|---:|---:|---:|---|---|
| 101 | 9.82944 | 10.50759 | 10.50759 | +0.67816 [0.5657777720665337, 0.7905330745521787] | [0.0, 1.0, 0.0, 0.0] |
| 102 | 9.72441 | 10.75297 | 10.75297 | +1.02856 [0.9115997921448666, 1.1455181336923534] | [1.0, 0.0, 0.0, 0.0] |
| 103 | 6.25322 | 10.79381 | 10.79381 | +4.54059 [4.407585076035269, 4.673603695531175] | [1.0, 0.0, 0.0, 0.0] |
| 104 | 10.41524 | 10.91506 | 10.91506 | +0.49982 [0.3909605202990794, 0.6086729002617907] | [0.0, 0.0, 0.0, 1.0] |
| 105 | 10.69964 | 10.84271 | 10.84271 | +0.14307 [0.03778261334988943, 0.24835312529377623] | [0.0, 0.0, 1.0, 0.0] |

Five-seed summaries (t intervals use independent training-seed means, not10000 trajectories as replicates):

```json
{
  "straight_through": {
    "training_seeds": 5,
    "mean": 10.762429065606097,
    "seed_sd": 0.15472400540811293,
    "minimum": 10.507591338985948,
    "maximum": 10.915061453309805,
    "seed_mean_t_ci95": [
      10.57031383843296,
      10.954544292779236
    ]
  },
  "original": {
    "training_seeds": 5,
    "mean": 9.384390395283408,
    "seed_sd": 1.796557313455477,
    "minimum": 6.253217049663595,
    "maximum": 10.699641150273893,
    "seed_mean_t_ci95": [
      7.1536698584560625,
      11.615110932110753
    ]
  },
  "paired_seed_differences": {
    "training_seeds": 5,
    "mean": 1.3780386703226912,
    "seed_sd": 1.796535413591561,
    "minimum": 0.14306786932183282,
    "maximum": 4.540594385783222,
    "seed_mean_t_ci95": [
      -0.8526546742320638,
      3.608732014877446
    ]
  },
  "positive_differences": 5,
  "one_expert_collapsed_seeds": 5,
  "inference_seed": 1001,
  "mc_uncertainty_is_not_seed_uncertainty": true
}
```

Stage frequencies exclude the separately learned first experiment. Whole-run/final20% exposure thresholds are norm>1e-8; near-zero<=1e-12. Full JSON reports router and encoder/expert gradients, actual parameter updates and selector saturation. Soft inference is a counterfactual policy; ST training forward is hard.

Observed expert collapse prevents interpreting an EIG improvement as evidence for useful four-expert specialization. One justified next controlled intervention is a temporary relaxed-forward expert-training warmup before returning to unchanged ST routing, targeting the documented unselected-expert starvation. Keep gamma/architecture/objective/optimizer/budget fixed and freeze one warmup duration before testing all original seeds; do not adopt or launch from this report.

Five original development seeds provide limited evidence and have already informed method development; they are not untouched confirmation. Reserved final evaluation seed8831001 remains unused.
