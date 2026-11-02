> **Budget dependency verified:** every arm remains1M-total. A provisional extra-calibration concern was rejected by source audit and an exact native-gradient test with NaN policy-target normalization. Only the frozen score model reads its preserved source calibration. See policy_normalization_dependency_test.json. Crossings remain diagnostics, not primary-result replacements.

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

## Frozen score-error diagnosis

16 saved histories: two most-boundary and two least-boundary cases for each crossing, both score bundles on identical histories. Native GPU normalization is loaded verbatim even though predictions run on CPU. No score or policy training. Fisher identity reference checked against finite differences; reliable only with ESS>=32 and <=20% score change from262144 to1M contrasts.

```json
{
  "status": "complete",
  "seconds": 98.91874850098975,
  "histories": 16,
  "model_predictions": 32,
  "new_primary_simulations": 0,
  "finite_difference_checks": [
    {
      "channel": "y",
      "finite_difference": 2.9173190922548993,
      "fisher_score": 2.917319092444816
    },
    {
      "channel": "x",
      "finite_difference": 1.475266229356009,
      "fisher_score": 1.4752662293262366
    }
  ],
  "coverage": [
    {
      "score_seed": 102,
      "policy_seed": 102,
      "mean_boundary_coordinate_fraction": 0.28294166666666665,
      "fraction_episodes_touching_boundary": 0.9575
    },
    {
      "score_seed": 103,
      "policy_seed": 102,
      "mean_boundary_coordinate_fraction": 0.0,
      "fraction_episodes_touching_boundary": 0.0
    },
    {
      "score_seed": 103,
      "policy_seed": 103,
      "mean_boundary_coordinate_fraction": 0.0,
      "fraction_episodes_touching_boundary": 0.0
    },
    {
      "score_seed": 102,
      "policy_seed": 103,
      "mean_boundary_coordinate_fraction": 0.445875,
      "fraction_episodes_touching_boundary": 0.9915
    }
  ],
  "summary": [
    {
      "score_bundle": 102,
      "stratum": "most_boundary",
      "cases": 8,
      "reliable_cases": 5,
      "reliable_mean_relative_error": 0.1554758870102291,
      "reliable_mean_design_cosine": 0.9345220844241116
    },
    {
      "score_bundle": 102,
      "stratum": "least_boundary",
      "cases": 8,
      "reliable_cases": 1,
      "reliable_mean_relative_error": 0.2960375039746806,
      "reliable_mean_design_cosine": 0.9555738768707427
    },
    {
      "score_bundle": 103,
      "stratum": "most_boundary",
      "cases": 8,
      "reliable_cases": 5,
      "reliable_mean_relative_error": 0.2194253431750794,
      "reliable_mean_design_cosine": 0.7983808810254797
    },
    {
      "score_bundle": 103,
      "stratum": "least_boundary",
      "cases": 8,
      "reliable_cases": 1,
      "reliable_mean_relative_error": 0.3701240476298259,
      "reliable_mean_design_cosine": 0.928703938359882
    }
  ],
  "reference": "Fisher identity explicit-likelihood prior importance, diagnostic only",
  "reliability_rule": "ESS>=32 and relative score change262144 to1M<=0.2",
  "limitation": "Extremal boundary strata are descriptive and not a representative population average. Low-ESS references cannot establish score error or its causal effect. No score-model hyperparameters changed."
}
```

A score-training correction is not identified solely by crossing or training score loss. Do not substitute a favorable frozen bundle into primary results. If trustworthy references show a consistent boundary-specific deficit, the next single controlled intervention is increased feasible-boundary coverage in the score-training design sampler at the same episode budget. Otherwise retain the unresolved mechanism and avoid selecting a speculative correction. This is a recommendation gate, not an implemented change.

## Next ScoreBED decision after reliability checks

No score-training correction is justified yet. Next diagnostic: compare both frozen bundles’ score-derived policy gradients at identical policy103 initialization on paired simulated histories against explicit-likelihood pathwise diagnostic gradients. Keep training and primary results unchanged; test gradient direction and scale before selecting a score-training correction.

Matched-history evidence: `{"reliable_boundary_histories": 4, "reliable_central_histories": 2, "boundary_mean_design_error_102": 0.3180995505697907, "boundary_mean_design_error_103": 0.5262253876081011, "boundary_mean_design_cosine_102": 0.9361312510355743, "boundary_mean_design_cosine_103": 0.7835288234338158, "central_mean_design_error_102": 0.3650859533841741, "central_mean_design_error_103": 0.4772098784414135}`. This descriptive screen does not establish causality. No corrective training was launched.

## Final ScoreBED root-cause assessment

Confirmed: bundle102 produces poor policies at both tested policy seeds;
bundle103 produces higher EIG with the same initialization/randomness within
each policy-seed comparison. The bundle-by-policy-seed interaction is also
large. This identifies a complete frozen-bundle contribution, including its
own normalization, rather than isolated weight causality or a universal
ranking of score-model seeds.

Boundary coordinates mean abs(design)>3.8. Bundle102 produces28.294% boundary
coordinates with policy102 and44.588% with policy103; bundle103 gives0% in
both. This is a behavioral association, not proof of a boundary score defect.

Only6/16 Fisher-reference histories pass ESS/resolution checks. On the four
reliable actual-boundary histories, bundle102 has mean design-score relative
error0.3181 and cosine0.9361, compared with bundle103's0.5262 and0.7835.
On the two reliable central histories, relative errors are0.3651 versus0.4772.
Thus these data do NOT support globally worse score accuracy for bundle102,
nor the proposed boundary-coverage training correction. The histories were
stratified extremes; these are not population-average accuracy estimates.

Unresolved: which early policy-gradient distortions or optimization paths
drive the boundary-attracted low-EIG outcome. One justified next diagnostic
is to compare both frozen bundles' score-derived policy gradients at identical
policy103 initialization, using paired histories and explicit-likelihood
pathwise diagnostic gradients. Test direction and scale before choosing a
score-training correction. No score-training modification is justified yet.
Original primary results remain unchanged; no favorable bundle is substituted.
