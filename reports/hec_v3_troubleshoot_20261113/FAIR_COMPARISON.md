# Exact original V2 experiment: V2 versus new four-pathway V3

**Same2000 cases,same generating truths/observation noise,same saved1M prior contrasts,sameT30 location-finding likelihood and bounds. No training.**

| Policy | Estimated EIG,nats |
|---|---:|
|V2|12.92935261|
|V2_uniform|12.93794410|
|V3|12.80310089|
|V3_uniform|12.80527233|
|expert1|12.73824955|
|expert2|12.85301397|
|expert3|12.86149725|
|expert4|12.81226900|

## Primary paired difference:V3 minus V2

```json
{
  "n": 2000,
  "mean": -0.12625171596406162,
  "MCSE": 0.025882726340085597,
  "CI95": [
    -0.17701166148380648,
    -0.07549177044431676
  ],
  "p_two_sided": 1.1573247658589832e-06
}
```

V2 learned and uniform saved per-case scores reproduce to maximum errors2.49e-14 and2.66e-14,respectively. This validates the original experimental bank and scorer.

Identical original V2 physical experiment,truth/noise cases,prior contrast bank,likelihood,bounds and horizon. Evaluation matched;V3 has additional training and compute,not training-budget matched to frozen V2.

Historical1M prior-sPCE is ceiling-limited;use16M paired results for better-resolved conclusions. Frozen checkpoint comparison,one lineage,no training.

Uniform semantics:V2 weights feature packets uniformly;V3 chooses one standalone expert uniformly. These are their respective architecture-consistent ablations,not identical policies. Single-expert intervals in the raw summary are exploratory/unadjusted.

Artifacts:`/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_same_V2_bank_20261113_02`
