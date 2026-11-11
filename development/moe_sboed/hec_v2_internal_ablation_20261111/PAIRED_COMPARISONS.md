# Paired full-minus-ablation comparisons

```json
{
  "expert1": {
    "n": 2000,
    "mean": 1.7361361031375488,
    "MCSE": 0.049748582068590355,
    "CI95": [
      1.638571600762261,
      1.8337006055128366
    ],
    "p_two_sided": 8.763315754120968e-209,
    "Holm5_p": 4.381657877060484e-208
  },
  "expert2": {
    "n": 2000,
    "mean": 0.9873371834104991,
    "MCSE": 0.0450018043571287,
    "CI95": [
      0.8990818308852784,
      1.07559253593572
    ],
    "p_two_sided": 8.939451594032398e-96,
    "Holm5_p": 2.6818354782097193e-95
  },
  "expert3": {
    "n": 2000,
    "mean": 1.5082854078786874,
    "MCSE": 0.048665948176545334,
    "CI95": [
      1.4128441144983086,
      1.6037267012590661
    ],
    "p_two_sided": 1.4749288473725862e-172,
    "Holm5_p": 5.899715389490345e-172
  },
  "expert4": {
    "n": 2000,
    "mean": 0.5015294790036194,
    "MCSE": 0.03832066373078289,
    "CI95": [
      0.42637685490611604,
      0.5766821031011228
    ],
    "p_two_sided": 1.2991492445982017e-37,
    "Holm5_p": 2.5982984891964033e-37
  },
  "uniform": {
    "n": 2000,
    "mean": -0.008591491661876416,
    "MCSE": 0.02738071107117092,
    "CI95": [
      -0.06228921206438307,
      0.04510622874063024
    ],
    "p_two_sided": 0.7537216472566813,
    "Holm5_p": 0.7537216472566813
  }
}
```

Five planned comparisons use Holm5;displayed95%tCIs are unadjusted. The best-observed-expert comparison belongs to this corrected family;its empirical winner label does not prove the population ranking.

Full minus compatible conventional direct-head MoE(secondary;raw and supplementary Holm6 p):

```json
{
  "n": 2000,
  "mean": 0.8088258315048582,
  "MCSE": 0.04927572039552068,
  "CI95": [
    0.7121886824712553,
    0.9054629805384611
  ],
  "p_two_sided": 6.723831297947295e-57,
  "supplementary_Holm6_p": 2.0171493893841884e-56
}
```
