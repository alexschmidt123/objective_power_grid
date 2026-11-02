# ScoreBED frozen-score crossed diagnosis

Ordered labels: score-bundle seed / policy-training seed. All complete models account for exactly1M primary episodes; reused score pretraining702994 plus new policy initialization10 and policy training296996. Final-budget checkpoints, common2000/L1M development1001.

| Score bundle | Policy seed | EIG | MC SE |
|---|---|---:|---:|
| 102 | 102 | 7.18907 | 0.06226 |
| 103 | 102 | 10.34629 | 0.05157 |
| 103 | 103 | 10.13136 | 0.05285 |
| 102 | 103 | 4.87567 | 0.06150 |

Paired contrasts:

```json
{
  "bundle103_minus102_policy102": {
    "difference": 3.1572229131780385,
    "mc_se": 0.07619775946140914,
    "ci95": [
      3.007875304633677,
      3.3065705217224
    ]
  },
  "bundle103_minus102_policy103": {
    "difference": 5.255683346088376,
    "mc_se": 0.0768302637162242,
    "ci95": [
      5.105096029204577,
      5.406270662972175
    ]
  },
  "bundle_by_policy_interaction": {
    "difference": 2.098460432910337,
    "mc_se": 0.09437737446068144,
    "ci95": [
      1.9134807789674015,
      2.2834400868532727
    ]
  }
}
```

Native policy103 initialization/RNG, normalization and frozen bundle102 prediction compatibility passed preflight. Historical103/103 is reused; no unnecessary new control training. This is a complete score-bundle intervention, not isolated score-weight causality. Score-error diagnostics and reliability filtering are reported separately.

Bundle103 outperforms102 at both policy initializations on these paired cases. This supports a score-bundle contribution; differing effect sizes can still indicate interaction. It does not isolate the responsible score-training defect.

No favorable score-bundle substitution into primary results. No targeted score-training correction is selected from crossing alone; require score-error evidence. Original five-seed results and reserved final evaluation seed8831001 remain untouched.
