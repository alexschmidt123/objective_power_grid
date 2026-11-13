# Frozen V3 diagnosis: routing quality or expert design

Status:**COMPLETE**

Only two categories:critic/selector/credit issues belong to routing quality;conditional expert complementarity/settings belong to expert design. Existing V3 experts differ by independent training seeds,not deliberately designed sBOED strategies.

## spce

```json
{
  "selection_chosen_experts": [
    2,
    3,
    2,
    3,
    1,
    2,
    1,
    1,
    1,
    3,
    3,
    3,
    1,
    1,
    2,
    3,
    1,
    3,
    3,
    3,
    1,
    1,
    3,
    3
  ],
  "critic_chosen_experts": [
    3,
    3,
    1,
    1,
    2,
    1,
    1,
    3,
    1,
    2,
    1,
    1,
    2,
    1,
    3,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    1,
    1
  ],
  "router_argmax_experts": [
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1,
    3,
    3,
    1,
    3
  ],
  "selection_chosen_fixed": 1,
  "primary_selection_choice_minus_learned": {
    "n": 24,
    "mean": -0.005981618771856064,
    "MCSE": 0.016226204902140454,
    "CI95": [
      -0.03954808103088779,
      0.027584843487175662
    ],
    "p_two_sided": 0.7157680474373016
  },
  "expert_design_conditional_minus_fixed": {
    "n": 24,
    "mean": -0.012214551355520001,
    "MCSE": 0.019995511830448044,
    "CI95": [
      -0.05357841907780047,
      0.02914931636676047
    ],
    "p_two_sided": 0.5472791892245797
  },
  "routing_critic_choice_minus_learned": {
    "n": 24,
    "mean": -0.005481342128141442,
    "MCSE": 0.012705987946037924,
    "CI95": [
      -0.03176568079060549,
      0.020802996534322604
    ],
    "p_two_sided": 0.6701954472999817
  },
  "routing_selection_choice_minus_critic_choice": {
    "n": 24,
    "mean": -0.0005002766437146219,
    "MCSE": 0.02051934012958649,
    "CI95": [
      -0.04294776576356071,
      0.04194721247613146
    ],
    "p_two_sided": 0.9807592101462154
  },
  "learned_minus_uniform": {
    "n": 24,
    "mean": 0.0009231683468717092,
    "MCSE": 0.007481902780502365,
    "CI95": [
      -0.01455432678042995,
      0.016400663474173367
    ],
    "p_two_sided": 0.9028726906011618
  },
  "critic_agreement_with_selection": 0.25,
  "router_agreement_with_selection": 0.3333333333333333,
  "test_mean_by_expert": [
    13.531292664678388,
    13.509361042895355,
    13.531755983669811
  ],
  "interpretation": "Expert design: independently validated conditional advantages. Routing quality: exploiting those advantages;critic ranking/selector optimization are subissues. Agreement alone is not a utility test."
}
```

## native_return

```json
{
  "selection_chosen_experts": [
    2,
    2,
    1,
    3,
    2,
    1,
    1,
    1,
    1,
    2,
    3,
    3,
    1,
    3,
    3,
    1,
    1,
    3,
    3,
    2,
    1,
    2,
    3,
    2
  ],
  "critic_chosen_experts": [
    3,
    3,
    1,
    1,
    2,
    1,
    1,
    3,
    1,
    2,
    1,
    1,
    2,
    1,
    3,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    1,
    1
  ],
  "router_argmax_experts": [
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1,
    3,
    3,
    1,
    3
  ],
  "selection_chosen_fixed": 3,
  "primary_selection_choice_minus_learned": {
    "n": 24,
    "mean": 0.04153586427370707,
    "MCSE": 0.024283540612880062,
    "CI95": [
      -0.00869846682304732,
      0.09177019537046147
    ],
    "p_two_sided": 0.10063946041333677
  },
  "expert_design_conditional_minus_fixed": {
    "n": 24,
    "mean": 0.022395985821882885,
    "MCSE": 0.021699024703657664,
    "CI95": [
      -0.022491866770009483,
      0.06728383841377525
    ],
    "p_two_sided": 0.3127477749774913
  },
  "routing_critic_choice_minus_learned": {
    "n": 24,
    "mean": -0.017211501797040302,
    "MCSE": 0.012547802868610243,
    "CI95": [
      -0.04316860969522885,
      0.008745606101148246
    ],
    "p_two_sided": 0.18339838770673522
  },
  "routing_selection_choice_minus_critic_choice": {
    "n": 24,
    "mean": 0.058747366070747375,
    "MCSE": 0.021886379850368703,
    "CI95": [
      0.013471939828760042,
      0.1040227923127347
    ],
    "p_two_sided": 0.013245019587445738
  },
  "learned_minus_uniform": {
    "n": 24,
    "mean": 0.0005959744254748026,
    "MCSE": 0.012282690327397932,
    "CI95": [
      -0.024812706396717364,
      0.026004655247666972
    ],
    "p_two_sided": 0.9617195277148148
  },
  "critic_agreement_with_selection": 0.3333333333333333,
  "router_agreement_with_selection": 0.25,
  "test_mean_by_expert": [
    5.175220489501953,
    5.205860137939453,
    5.220144748687744
  ],
  "interpretation": "Expert design: independently validated conditional advantages. Routing quality: exploiting those advantages;critic ranking/selector optimization are subissues. Agreement alone is not a utility test."
}
```

## Limitations

24 diagnostic histories,finite importance posterior and contrast bounds,one checkpoint. Single initial decision only;no deployable oracle claim. Independent banks validate selection;rank disagreement alone is not a bug. Secondary comparisons exploratory/unadjusted. Existing V3 expert settings differ only by training seeds,not designed sBOED strategies.

Artifacts:`/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_routing_or_experts_20261113`
