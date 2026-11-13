# HEC-v2 frozen-head router refinement

One change: train only the existing coordinator and response gate while fixing all other actor parameters. Native critic/posterior updates and information objective are unchanged.

Artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v2_router_only_20261112`

Each arm completed 2,000 updates from the same iteration-16001 runtime. Primary evaluation uses 2,048 fresh episodes and 16 million prior contrasts. Final checkpoints only; no test-based selection.

Measured paired comparisons (native evaluator result fields):

## primary_router_only_minus_ordinary

```json
{
  "n": 2048,
  "mean": -0.03349347405087735,
  "MCSE": 0.040894948939415116,
  "CI95": [
    -0.11369352189537447,
    0.04670657379361977
  ],
  "p_two_sided": 0.41287462485314036
}
```

## secondary_router_only_minus_original

```json
{
  "n": 2048,
  "mean": 0.034814973539646486,
  "MCSE": 0.02949564949173137,
  "CI95": [
    -0.02302963959206037,
    0.09265958667135335
  ],
  "p_two_sided": 0.23800109489083537
}
```

## secondary_router_only_minus_own_uniform

```json
{
  "n": 2048,
  "mean": 0.012733331161576822,
  "MCSE": 0.03960390026340408,
  "CI95": [
    -0.06493481070746462,
    0.09040147303061827
  ],
  "p_two_sided": 0.7478513812900589
}
```

## secondary_router_only_minus_original_uniform

```json
{
  "n": 2048,
  "mean": 0.012733331161576822,
  "MCSE": 0.03960390026340408,
  "CI95": [
    -0.06493481070746462,
    0.09040147303061827
  ],
  "p_two_sided": 0.7478513812900589
}
```

## Limitations

One inherited training lineage; 16M prior-sPCE is a finite-contrast lower bound. Secondary intervals unadjusted.

A gain is supported only if the primary paired confidence interval excludes zero in the positive direction. A null or negative result rejects this intervention as an improvement on this lineage; it does not prove routing impossible.

Frozen actor parameters and buffers were verified unchanged in both arms; historical source/checkpoint hashes were verified after evaluation.
