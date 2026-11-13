# Matched V2 E4 adapter refresh

Accepted under locked joint competency/EIG rule: **False**

Artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v2_E4_refresh_20261113`

## competency

```json
{
  "cases": 256,
  "teacher": "frozen original V2",
  "primary_ordinary_minus_refresh_error": {
    "n": 256,
    "mean": -0.0005001280726222815,
    "MCSE": 0.0003973310791631386,
    "CI95": [
      -0.0012825963665481078,
      0.0002823402213035449
    ],
    "p_two_sided": 0.20928307776566937
  },
  "original_minus_refresh_error": {
    "n": 256,
    "mean": -0.00026481218804940223,
    "MCSE": 0.0007258463513087155,
    "CI95": [
      -0.0016942290797686212,
      0.001164604703669817
    ],
    "p_two_sided": 0.7155389844820756
  },
  "errors": {
    "original": 0.17089295531978785,
    "ordinary": 0.17065763943521498,
    "E4_refresh": 0.17115776750783726
  },
  "targets_are_model_derived": true
}
```

## EIG

```json
{
  "primary_refresh_minus_ordinary": {
    "n": 2048,
    "mean": -0.009966169024988585,
    "MCSE": 0.03781705702718742,
    "CI95": [
      -0.084130090525703,
      0.06419775247572584
    ],
    "p_two_sided": 0.7921637382253213
  },
  "secondary_refresh_minus_original": {
    "n": 2048,
    "mean": -0.003662278380957843,
    "MCSE": 0.04272343493698768,
    "CI95": [
      -0.08744821319417483,
      0.08012365643225915
    ],
    "p_two_sided": 0.9316969108554647
  },
  "secondary_refresh_minus_uniform": {
    "n": 2048,
    "mean": 0.19295586484661886,
    "MCSE": 0.044533660521569214,
    "CI95": [
      0.10561985399081157,
      0.28029187570242614
    ],
    "p_two_sided": 1.5434327571894307e-05
  },
  "means": {
    "original": 14.130905219830767,
    "ordinary": 14.137209110474796,
    "E4_refresh": 14.127242941449808,
    "E4_refresh_uniform": 13.934287076603189
  },
  "cases": 2048,
  "contrasts": 16000000
}
```

## Limitations

One training lineage;finite prior-sPCE and model-derived competency;extra adapter-backprop compute differs from readout-only control,reported explicitly. Audit test histories informed method choice and are not final evaluation.

Both arms:300 preparation steps and2000 unchanged native EIG updates. Control preparation updates only an unused diagnostic readout; candidate additionally updates adapter4. Additional adapter-backprop FLOPs are not exactly matched. Historical artifacts preserved.
