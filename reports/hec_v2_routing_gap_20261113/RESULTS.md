# Frozen V2 routing opportunity diagnostic

Status: **COMPLETE**

Artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v2_routing_gap_20261113`

No training. One routing intervention per conditional future; all later decisions use original V2. Selection and test use independent posterior/noise/contrast banks.

## primary_selected_minus_learned

```json
{
  "n": 24,
  "mean": 0.014764699519051053,
  "MCSE": 0.025776713390835267,
  "CI95": [
    -0.038558494808490926,
    0.06808789384659303
  ],
  "p_two_sided": 0.5723431044097715
}
```

## secondary_selected_minus_fixed

```json
{
  "n": 24,
  "mean": 0.04197938762858152,
  "MCSE": 0.022391740755154797,
  "CI95": [
    -0.0043414572950998265,
    0.08830023255226287
  ],
  "p_two_sided": 0.07357969872793599
}
```

## secondary_selected_minus_stage

```json
{
  "n": 24,
  "mean": 0.008166587850129284,
  "MCSE": 0.02092306163001983,
  "CI95": [
    -0.035116062824078025,
    0.05144923852433659
  ],
  "p_two_sided": 0.6998950294515496
}
```

## secondary_selected_minus_uniform

```json
{
  "n": 24,
  "mean": 0.017340132427244043,
  "MCSE": 0.016684063415508876,
  "CI95": [
    -0.01717348232996242,
    0.0518537471844505
  ],
  "p_two_sided": 0.30945770812457823
}
```

## Limitations

Finite importance posterior approximation and finite prior-sPCE; 24 histories, one checkpoint, six mixtures, one-decision interventions only. No deployable selector or repeated-routing gain established. Secondary intervals exploratory, unadjusted.

Posterior samples are finite importance approximations. ESS does not establish complete mode coverage. Finite prior-sPCE is not exact EIG. No deployable history selector or repeated-routing improvement was trained or tested.
