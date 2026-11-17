# V4 seed101 pilot results

| Method | EIG (nats) | MCSE |
|---|---:|---:|
| MoE-sBOED-V4 | 12.88401 | 0.03678 |
| MoE-sBOED-V4-uniform | 12.78326 | 0.03800 |
| Previous MoE-sBOED | 12.93099 | 0.03547 |
| Previous MoE-sBOED-uniform | 12.97854 | 0.03447 |

Paired V4 differences:

```json
{
  "MoE-sBOED-V4-uniform": {
    "n": 2000,
    "mean": 0.100743303935264,
    "MCSE": 0.030504077542369737,
    "CI95": [
      0.04092018894205266,
      0.16056641892847534
    ]
  },
  "Previous MoE-sBOED": {
    "n": 2000,
    "mean": -0.04698740997050626,
    "MCSE": 0.03360155661760538,
    "CI95": [
      -0.11288515045106456,
      0.018910330510052052
    ]
  },
  "Previous MoE-sBOED-uniform": {
    "n": 2000,
    "mean": -0.0945351388868159,
    "MCSE": 0.03428969148204251,
    "CI95": [
      -0.16178241603403398,
      -0.027287861739597838
    ]
  }
}
```

Five-seed recommendation:NO,predeclared pilot criteria not met

One paired continuation seed on development-validation bank;case MC uncertainty is not across-seed robustness. No exact-MI or from-scratch compute superiority claim.
