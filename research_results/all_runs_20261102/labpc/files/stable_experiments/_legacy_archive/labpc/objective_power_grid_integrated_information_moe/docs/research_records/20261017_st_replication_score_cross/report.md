# MoE straight-through replication and ScoreBED crossed-score investigation

Two separate location-finding development investigations. Every complete model is exactly1M-total. Final-budget checkpoints,2000 paired held-out trajectories,L1M,development seed1001. Original five-seed results are preserved. Final evaluation seed8831001 was not sampled. Score crossings are diagnostic interventions, never primary-result replacements.

## A. Straight-through MoE: all five seeds

| Training seed | Original hard EIG | ST hard EIG | ST relaxed EIG | Gain | E1/E2/E3/E4 usage |
|---|---:|---:|---:|---:|---|
| 101 | 9.82944 | 10.50759 | 10.50759 | +0.67816 | 0.0%/100.0%/0.0%/0.0% |
| 102 | 9.72441 | 10.75297 | 10.75297 | +1.02856 | 100.0%/0.0%/0.0%/0.0% |
| 103 | 6.25322 | 10.79381 | 10.79381 | +4.54059 | 100.0%/0.0%/0.0%/0.0% |
| 104 | 10.41524 | 10.91506 | 10.91506 | +0.49982 | 0.0%/0.0%/0.0%/100.0% |
| 105 | 10.69964 | 10.84271 | 10.84271 | +0.14307 | 0.0%/0.0%/100.0%/0.0% |

ST mean **10.76243**, seed SD **0.15472**, minimum **10.50759**. Original mean 9.38439, SD 1.79656, minimum 6.25322.

Mean paired seed gain +1.37804, five-training-seed t interval [-0.8526546742320638, 3.608732014877446]. Per-trajectory MC intervals are separately saved in moe_results.json; do not treat10000 trajectories as10000 independent trained policies.

Positive point differences in 5/5 seeds; >=99.9% final single-expert selection in 5/5 seeds.

The four64/64 tanh experts, sum encoder, native DSelect-1, gamma30%/95%/.001, optimizer, sPCE L1024, regularization and clipping were unchanged. Seeds103/105 are the preserved completed runs;101/102/104 are newly trained. ST forward is hard; the relaxed evaluation is a counterfactual, not the ST training forward.

### Gradient exposure

| Seed | Whole-run E1/E2/E3/E4 exposure | Final20% router exposure | Final20% selector saturation |
|---|---|---:|---:|
| 101 | 0.06%/100.00%/0.00%/0.00% | 0.00% | 100.00% |
| 102 | 100.00%/0.00%/0.03%/0.00% | 0.00% | 100.00% |
| 103 | 100.00%/0.00%/0.00%/0.00% | 0.00% | 100.00% |
| 104 | 0.00%/0.04%/0.00%/100.00% | 0.00% | 100.00% |
| 105 | 0.00%/0.00%/100.00%/0.00% | 0.00% | 100.00% |

Gradient exposure means norm>1e-8. One-expert collapse is a negative mechanism result even if held-out EIG improves. The evidence cannot attribute a gain to useful four-expert specialization. A justified next controlled development intervention is temporary relaxed-forward expert warmup before unchanged ST routing, targeting undertrained experts. Freeze one duration, retain the original gamma schedule and episode budget, and test all five original seeds; this intervention is not implemented or launched.

## B. ScoreBED: crossed frozen score bundles

| Score bundle seed | Policy training seed | Held-out EIG | MC SE |
|---|---|---:|---:|
| 102 | 102 | 7.18907 | 0.06226 |
| 103 | 102 | 10.34629 | 0.05157 |
| 103 | 103 | 10.13136 | 0.05285 |
| 102 | 103 | 4.87567 | 0.06150 |

bundle103_minus102_policy102: +3.15722; paired95% MC interval [3.007875304633677, 3.3065705217224].

bundle103_minus102_policy103: +5.25568; paired95% MC interval [5.105096029204577, 5.406270662972175].

bundle_by_policy_interaction: +2.09846; paired95% MC interval [1.9134807789674015, 2.2834400868532727].

The missing crossing changes the frozen score bundle relative to native policy103: policy initialization/RNG follows103, score configuration/variables/normalization follows102. Initial parameters and keys match the native loader; exact bundle102 score predictions passed preflight. Both bundles were evaluated on identical physical histories. The intervention identifies a complete bundle effect and interaction, not isolated score-weight or source-normalization causality. **Budget audit:** every arm remains exactly1M-total. Policy-target normalization fields are unused by native DADNet initialization, physical simulation and score-derived gradients; replacing them with NaNs leaves initial parameters and native gradients bitwise identical. Only the frozen score network consumes its source calibration, already charged within702994 reused pretraining episodes. A provisional1.01M concern was rejected by this test and static call-path audit; all records are preserved. New policy initialization/training is297006 episodes, and no second calibration dataset is needed.

Poor results with bundle102 at both policy seeds support a score-bundle contribution. Two policy initializations cannot establish a universal bundle ranking. Score-error references use Fisher identity, finite-difference verification, ESS>=32 and <=20% resolution change262k→1M. Low-ESS cases cannot support claims of score inaccuracy.

Score-error summary:

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

No automatic score-training correction is justified by EIG crossing alone. If reliable matched-history errors establish a boundary-specific defect, the single candidate intervention is improved feasible-boundary coverage of score-training designs at the unchanged total episode budget. Otherwise recommend a matched policy-gradient error diagnostic before altering score training. Do not substitute bundle103 into primary seed102.

## Cost and preservation

New training:3M MoE primary episodes plus297006 new ScoreBED initialization/policy episodes. The ScoreBED complete-model budget includes702994 reused score-pretraining episodes. Reused MoE103/105 are not retrained. Correctness probes46 simulation episodes and validation/evaluation overhead are separate; no preflight weights are used for training. GPU phases are serialized. A14.43-second CPU-only dependency probe on two pinned cores overlapped seed102; retain this provenance rather than claiming perfectly identical wall-clock conditions. Measured root-phase wall times are in completion/, native optimization/validation times in each training record. Score pretraining wall time was not separately saved historically and is not invented.

All original checkpoint hashes, both prior ST checkpoints, new frozen scientific sources and historical seed101 alias checkpoint were verified intact. Execution bookkeeping fixes (output path rejected before training, historical seed101 registry resolution, and a coordinator filename that shadowed the standard-library queue module during diagnostics) are preserved with provenance; none changed scientific training settings. The provisional extra-calibration concern was rejected by a native-gradient dependency test, with both initial and provisional records retained.

Machine-readable results: moe_results.csv/.json, moe_diagnostics.json, score_cross_results.csv/.json, score_error_records.json and score_error_references.json. Exact commands/configurations/source and checkpoint hashes are retained in the campaign.

![Investigation results](investigation_results.png)

## Retained late-checkpoint behavior

The full saved512-case/L1M validation curves use development771001 and are separate from the2000-case final development1001 scores. No intermediate checkpoint is selected. The sustained-drop screen is exploratory, unadjusted for repeated looks, and may identify temporary fluctuations rather than terminal failure.

```json
{
  "101": {
    "late_sustained_drop": null,
    "final_minus_750016": {
      "difference": -0.0927954310791547,
      "mc_se": 0.09116219303584015,
      "ci95": [
        -0.2714733294294014,
        0.085882467271092
      ]
    },
    "final_minus_800000": {
      "difference": -0.13952303050003034,
      "mc_se": 0.08617490487271208,
      "ci95": [
        -0.308425844050546,
        0.02937978305048533
      ]
    }
  },
  "102": {
    "late_sustained_drop": null,
    "final_minus_750016": {
      "difference": 0.04849951563710267,
      "mc_se": 0.07497566472824174,
      "ci95": [
        -0.09845278723025114,
        0.19545181850445648
      ]
    },
    "final_minus_800000": {
      "difference": -0.04386218077803926,
      "mc_se": 0.07901616375873222,
      "ci95": [
        -0.1987338617451544,
        0.11100950018907589
      ]
    }
  },
  "103": {
    "late_sustained_drop": null,
    "final_minus_750016": {
      "difference": -0.04256179779766591,
      "mc_se": 0.08765887954123718,
      "ci95": [
        -0.21437320169849078,
        0.12924960610315894
      ]
    },
    "final_minus_800000": {
      "difference": 0.12781216780331658,
      "mc_se": 0.0729000959465989,
      "ci95": [
        -0.01507202025201726,
        0.27069635585865043
      ]
    }
  },
  "104": {
    "late_sustained_drop": null,
    "final_minus_750016": {
      "difference": 0.2310223535345438,
      "mc_se": 0.09252751818290746,
      "ci95": [
        0.0496684178960452,
        0.4123762891730424
      ]
    },
    "final_minus_800000": {
      "difference": 0.028908248202706485,
      "mc_se": 0.078043834825435,
      "ci95": [
        -0.12405766805514612,
        0.1818741644605591
      ]
    }
  },
  "105": {
    "late_sustained_drop": {
      "first_checkpoint": 825024,
      "anchor": 750016,
      "paired_changes": [
        {
          "difference": -0.3529424936681532,
          "mc_se": 0.09269974334822585,
          "ci95": [
            -0.5346339906306758,
            -0.17125099670563054
          ]
        },
        {
          "difference": -0.549540078579985,
          "mc_se": 0.10423266796107035,
          "ci95": [
            -0.7538361077836828,
            -0.3452440493762871
          ]
        },
        {
          "difference": -0.3493296629595471,
          "mc_se": 0.09166527381924917,
          "ci95": [
            -0.5289935996452755,
            -0.1696657262738187
          ]
        }
      ]
    },
    "final_minus_750016": {
      "difference": -0.06292910559661283,
      "mc_se": 0.0901722196621454,
      "ci95": [
        -0.23966665613441782,
        0.11380844494119216
      ]
    },
    "final_minus_800000": {
      "difference": 0.41510568577940177,
      "mc_se": 0.0931350710356701,
      "ci95": [
        0.23256094654948836,
        0.5976504250093152
      ]
    }
  }
}
```

![Late validation curves](late_validation_curves.png)

## Next ScoreBED decision after reliability checks

No score-training correction is justified yet. Next diagnostic: compare both frozen bundles’ score-derived policy gradients at identical policy103 initialization on paired simulated histories against explicit-likelihood pathwise diagnostic gradients. Keep training and primary results unchanged; test gradient direction and scale before selecting a score-training correction.

Matched-history evidence: `{"reliable_boundary_histories": 4, "reliable_central_histories": 2, "boundary_mean_design_error_102": 0.3180995505697907, "boundary_mean_design_error_103": 0.5262253876081011, "boundary_mean_design_cosine_102": 0.9361312510355743, "boundary_mean_design_cosine_103": 0.7835288234338158, "central_mean_design_error_102": 0.3650859533841741, "central_mean_design_error_103": 0.4772098784414135}`. This descriptive screen does not establish causality. No corrective training was launched.

## Final MoE interpretation

All five development-seed point differences are positive, with improved observed
mean, SD and minimum. However, the paired five-training-seed95% t interval for
the mean gain is[-0.85265,3.60873] nats, so it does not establish population
superiority. These are development seeds, not untouched confirmation.

Every final policy selects one expert for all58000 post-initial decisions.
Its dominant expert receives meaningful gradients on100% of updates. Other
experts receive at most9/15625 updates (0.0576%), with most receiving none;
the exact per-expert counts/fractions remain in moe_diagnostics.json. During
the final20%, every run has zero router-gradient exposure,100% selector
saturation and zero non-dominant-expert exposure. Hard/relaxed final scores
are identical case by case. This is strong single-active-expert performance,
not demonstrated useful four-expert specialization.

The next single intervention is1563 batches (100032 episodes) of native relaxed
expert warmup, followed by14062 batches (899968 episodes) of unchanged ST.
The total remains1M, with the original gamma schedule and all other settings.
This is the batch-exact version of the nominal10% warmup recommendation;
it is not implemented or launched.

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
