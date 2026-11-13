# Frozen V2 expert contribution diagnostic

Status: **COMPLETE**

Artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v2_expert_contribution_20261113_posterior_refined`

No training. Single-decision and persistent all-consumer expert removals. History regimes are fitted on selection histories and tested on independent histories and simulation banks.

## primary_conditional_minus_global_damage

```json
{
  "n": 24,
  "mean": 0.0,
  "MCSE": 0.0,
  "CI95": [
    0.0,
    0.0
  ],
  "p_two_sided": 1.0
}
```

## regime

```json
{
  "feature": 0,
  "threshold": 0.32724137931034514,
  "labels": [
    2,
    2
  ],
  "selection_mean_damage": 0.2636968445542428,
  "global_expert": 2
}
```

## test_regimes

```json
{
  "0": {
    "histories": 8,
    "selected_expert": 3,
    "mean_persistent_damage": [
      0.14108808505321213,
      0.21420888302060295,
      0.37637212604681386,
      0.19782665673868177
    ]
  },
  "1": {
    "histories": 16,
    "selected_expert": 3,
    "mean_persistent_damage": [
      0.060055709609875906,
      -0.04367835877053954,
      0.2408425938354955,
      -0.07441365352843153
    ]
  }
}
```

## expert_removal_effects

```json
{
  "single_E1": {
    "n": 24,
    "mean": 0.007583188941695962,
    "MCSE": 0.012703165438331158,
    "CI95": [
      -0.01869531091872002,
      0.03386168880211195
    ],
    "p_two_sided": 0.5563703015331942,
    "p_Holm8": 1
  },
  "single_E2": {
    "n": 24,
    "mean": -0.01132099960035012,
    "MCSE": 0.010596866848231135,
    "CI95": [
      -0.033242288852540776,
      0.010600289651840534
    ],
    "p_two_sided": 0.2964547600489111,
    "p_Holm8": 1
  },
  "single_E3": {
    "n": 24,
    "mean": -0.026874265890208287,
    "MCSE": 0.019802397286019055,
    "CI95": [
      -0.06783864574047312,
      0.014090113960056546
    ],
    "p_two_sided": 0.18791233975059018,
    "p_Holm8": 1
  },
  "single_E4": {
    "n": 24,
    "mean": -0.013755791348118848,
    "MCSE": 0.01648354281895098,
    "CI95": [
      -0.047854597647210055,
      0.020343014950972356
    ],
    "p_two_sided": 0.4125741997894118,
    "p_Holm8": 1
  },
  "persistent_E1": {
    "n": 24,
    "mean": 0.08706650142432126,
    "MCSE": 0.032460555614525564,
    "CI95": [
      0.01991672601390218,
      0.15421627683474035
    ],
    "p_two_sided": 0.013304317310494296,
    "p_Holm8": 0.09313022117346008
  },
  "persistent_E2": {
    "n": 24,
    "mean": 0.042284055159841204,
    "MCSE": 0.045170597744410505,
    "CI95": [
      -0.051158445631311104,
      0.1357265559509935
    ],
    "p_two_sided": 0.35894898248057816,
    "p_Holm8": 1
  },
  "persistent_E3": {
    "n": 24,
    "mean": 0.28601910457260143,
    "MCSE": 0.06723547934987169,
    "CI95": [
      0.1469319185253166,
      0.42510629061988625
    ],
    "p_two_sided": 0.0002989759938861823,
    "p_Holm8": 0.002391807951089458
  },
  "persistent_E4": {
    "n": 24,
    "mean": 0.016333116560605987,
    "MCSE": 0.06454447934509103,
    "CI95": [
      -0.11718731184715164,
      0.14985354496836362
    ],
    "p_two_sided": 0.8024760156258247,
    "p_Holm8": 1
  }
}
```

## Limitations

One checkpoint,24 selection and24 independent test histories,approximate importance posterior,finite prior-sPCE. Masking is an out-of-distribution intervention and can expose shared-head coadaptation; damage is not proof of intrinsic scientific specialization. Shared history/SG paths remain. Experts still execute; no compute saving. Secondary tests Holm8; regime leaf summaries descriptive.

Posterior samples are finite importance approximations. ESS does not establish complete mode coverage. Finite prior-sPCE is not exact EIG. No deployable history selector or repeated-routing improvement was trained or tested.
