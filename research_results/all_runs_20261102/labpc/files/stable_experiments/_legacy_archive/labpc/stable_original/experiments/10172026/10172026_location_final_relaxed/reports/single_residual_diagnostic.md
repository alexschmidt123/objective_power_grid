# Fixed dominant residual expert diagnostic

Same trained checkpoints, only development-selected dominant expert forced after the first design. No training and no new named method. Forcing can create out-of-distribution histories; this does not replace an independently trained single-residual control or prove useful specialization.

```json
[
  {
    "seed": 101,
    "fixed_expert": "E3",
    "fixed_eig": 10.595483560487361,
    "learned_eig": 10.563423685902304,
    "learned_minus_fixed": -0.03205987458505723,
    "paired_mc95_low": -0.07391548094463124,
    "paired_mc95_high": 0.009795731774516779,
    "diagnostic_only": true
  },
  {
    "seed": 102,
    "fixed_expert": "E1",
    "fixed_eig": 10.428559857406952,
    "learned_eig": 10.428559857406952,
    "learned_minus_fixed": 0.0,
    "paired_mc95_low": 0.0,
    "paired_mc95_high": 0.0,
    "diagnostic_only": true
  },
  {
    "seed": 103,
    "fixed_expert": "E1",
    "fixed_eig": 10.66521559033978,
    "learned_eig": 10.666717154525665,
    "learned_minus_fixed": 0.0015015641858842318,
    "paired_mc95_low": -0.00039221379000572553,
    "paired_mc95_high": 0.003395342161774189,
    "diagnostic_only": true
  },
  {
    "seed": 104,
    "fixed_expert": "E4",
    "fixed_eig": 10.407206520383191,
    "learned_eig": 10.486206696035028,
    "learned_minus_fixed": 0.07900017565183517,
    "paired_mc95_low": 0.009922352683711791,
    "paired_mc95_high": 0.14807799861995857,
    "diagnostic_only": true
  },
  {
    "seed": 105,
    "fixed_expert": "E2",
    "fixed_eig": 10.622501679954858,
    "learned_eig": 10.622718038586816,
    "learned_minus_fixed": 0.00021635863195836703,
    "paired_mc95_low": -0.0008726652907163172,
    "paired_mc95_high": 0.0013053825546330512,
    "diagnostic_only": true
  }
]
```
