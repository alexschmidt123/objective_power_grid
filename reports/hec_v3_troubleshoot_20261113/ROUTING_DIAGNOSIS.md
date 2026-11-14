# Routing quality versus expert design: frozen diagnostics

## Routing quality

No implementation bug found in the tested actor loss/gradient/optimizer/freeze paths. The actual actor loss equals the explicit categorical expected native-Q loss;logit gradients match the analytic softmax plus tanh-cap derivative. Every router tensor belongs to the optimizer with250 Adam steps and finite gradients. Saved expert weights and selector-initial state reproduce exactly. These tests do not prove all code is bug-free.

On24 fresh diagnostic histories,average E4 probability rises from0.48244 initially to0.91954 after training. The critic argmax chooses E4 in7/24 histories,and E2 in8/24. The cap derivative averages0.94983;there is no evidence of complete tanh-gradient shutdown. Probabilities/rankings alone are not utility evidence.

## Independent native-return continuation test

```json
{
  "selection_chosen": [
    1,
    2,
    3,
    4,
    1,
    3,
    3,
    3,
    3,
    4,
    1,
    2,
    3,
    1,
    1,
    1,
    1,
    1,
    3,
    3,
    2,
    1,
    2,
    3
  ],
  "critic_chosen": [
    1,
    2,
    2,
    4,
    1,
    3,
    1,
    3,
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    4,
    1,
    4,
    3,
    4,
    3,
    2,
    3,
    2
  ],
  "fixed_choice": 3,
  "independent_choice_minus_learned": {
    "n": 24,
    "mean": -0.0017247187594572704,
    "MCSE": 0.018765890964216266,
    "CI95": [
      -0.040544921918877305,
      0.03709548439996277
    ],
    "p_two_sided": 0.9275676617068932
  },
  "independent_choice_minus_fixed": {
    "n": 24,
    "mean": 0.00374476487437884,
    "MCSE": 0.020074694772087915,
    "CI95": [
      -0.03778290524274032,
      0.045272434991497995
    ],
    "p_two_sided": 0.8536556905630079
  },
  "independent_choice_minus_critic": {
    "n": 24,
    "mean": 0.023789444317420323,
    "MCSE": 0.011133649791901052,
    "CI95": [
      0.0007577349436637543,
      0.04682115369117689
    ],
    "p_two_sided": 0.04348350537834884
  },
  "critic_choice_minus_learned": {
    "n": 24,
    "mean": -0.025514163076877594,
    "MCSE": 0.019544777987563742,
    "CI95": [
      -0.06594561680480203,
      0.014917290651046841
    ],
    "p_two_sided": 0.20465793642382363
  },
  "average_native_return_by_expert": [
    4.860055446624756,
    4.833190441131592,
    4.841723918914795,
    4.847217082977295
  ],
  "critic_selection_agreement": 0.375,
  "note": "Native variational return,not16M prior-sPCE. Independent banks for choice/test;24 histories,approximate conditional posterior;secondary comparisons descriptive."
}
```

The independent-bank selected action does NOT significantly beat learned routing on native variational return. Its comparison with critic argmax is only nominally positive (p0.0435),exploratory/unadjusted;this is insufficient to confirm critic misranking as the root cause. This criterion is native terminal-minus-prefix logposterior,not16M prior-sPCE.

## Expert design

The four query mechanisms are distinct and their saved standalone weights remain intact. Distinct computations do not establish useful conditional complementarity. Conditional selection versus the selection-chosen fixed expert has CI[-0.03778,0.04527];this small diagnostic does not establish a conditional advantage. It also does not establish absence of one.

## Evidence-grounded next test

Do not launch another expert rewrite or longer router training from these results. The next targeted routing diagnostic should compare one-step expert selection with commitment to the same expert for the remaining horizon,using paired posterior-conditioned futures. The current one-step test returns to the learned continuation,whereas full-episode standalone policies change all future decisions. This distinction matters in sBOED and is unresolved here.

Artifacts:`/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_troubleshoot_native_20261113`
