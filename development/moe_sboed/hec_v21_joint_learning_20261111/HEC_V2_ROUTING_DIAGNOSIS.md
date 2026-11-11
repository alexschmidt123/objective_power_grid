# Verified HEC-v2 routing diagnosis

Four heterogeneous expert pathways produce128-coordinate packets(first64 plus response64),not complete actions. E1 retains a frozen geometry anchor plus live differences;E2 combines likelihood representations with exact-hypothesis semantics and52 learned responses;E3 retains a frozen sensitivity anchor plus live differences;E4 combines opportunity representations with predictive continuation semantics and52 learned responses. A shared history path and frozen SG reference remain active in every policy. These inherited frozen weights are preserved exactly;live specialists remain EIG trainable. No new hand-assigned experimental stages.

The20-dimensional commander affects real model queries. Old coordination is a nonlinear network on history/legacy representations plus a linear response gate,followed by bounded logits and softmax. Its native gradients are NONZERO. The audit finds mean logit-squash derivatives about.99,so saturation/dead gradients are not supported explanations. Within-stage gate variability is also nonzero. Raw gradient norms of different-sized parameter blocks are not directly comparable learning rates.

In192 saved histories at stages3/12/24,shared-history hidden contributions exceed the weighted expert hidden contribution by roughly2.15/3.37/4.15 in norm. The gate-to-action tangent RMS is about.036-.039 physical units per logit. Expert contribution scales differ after the actual fusion matrices. Cancellation ratios are about.65-.71;this is not proof of harmful redundancy. Actual aggregate EIG evidence shows full versus uniform unresolved,while one-hot expert pathways are substantially worse. It does not prove each expert indispensable or dynamically allocated by comparative information value.

Feature/head scale compensation is architecturally possible and plausible,not causally established. A fixed positive gate can be absorbed into expert fusion-column scales;history-dependent gates do not in general admit one exact fixed rescaling. The head can use history directly. Current raw-representation coordination is not explicitly informed of these consumed contribution scales or their physical action/likelihood effects. This motivates one candidate;it is not a proven cause or guaranteed cure.

Actual frozen audit(critic gradients at saved histories are NOT policy EIG):

```json
{
  "status": "COMPLETE_FROZEN_AUDIT",
  "cases": 192,
  "stages": {
    "3": {
      "gate_mean": [
        0.19431397318840027,
        0.3360034227371216,
        0.09227451682090759,
        0.37740808725357056
      ],
      "within_stage_gate_SD": [
        0.03313126042485237,
        0.07173901796340942,
        0.04519304633140564,
        0.06337326020002365
      ],
      "entropy": 1.2452713251113892,
      "raw_logit_tanh_derivative_mean": 0.9950042963027954,
      "decision_tangent_RMS": 0.037404611706733704,
      "contribution_norms": [
        2.5469226837158203,
        2.6474552154541016,
        6.548854827880859,
        1.1856716871261597
      ],
      "weighted_cancellation_ratio": 0.6492197513580322,
      "shared_history_to_expert_norm_ratio": 2.148777961730957,
      "physical_bound_fraction": 0.0
    },
    "12": {
      "gate_mean": [
        0.10944479703903198,
        0.3015871047973633,
        0.07256662100553513,
        0.516401469707489
      ],
      "within_stage_gate_SD": [
        0.038476139307022095,
        0.06863299757242203,
        0.042176906019449234,
        0.08078864961862564
      ],
      "entropy": 1.1039965152740479,
      "raw_logit_tanh_derivative_mean": 0.9910390377044678,
      "decision_tangent_RMS": 0.03913978859782219,
      "contribution_norms": [
        3.196986675262451,
        2.1903443336486816,
        12.434126853942871,
        1.0942325592041016
      ],
      "weighted_cancellation_ratio": 0.6742554903030396,
      "shared_history_to_expert_norm_ratio": 3.3678183555603027,
      "physical_bound_fraction": 0.0
    },
    "24": {
      "gate_mean": [
        0.05462292954325676,
        0.27266716957092285,
        0.18285509943962097,
        0.48985475301742554
      ],
      "within_stage_gate_SD": [
        0.02992754802107811,
        0.06746519356966019,
        0.07866162806749344,
        0.10207469016313553
      ],
      "entropy": 1.1303664445877075,
      "raw_logit_tanh_derivative_mean": 0.9915055632591248,
      "decision_tangent_RMS": 0.03592585399746895,
      "contribution_norms": [
        4.736480712890625,
        2.524486541748047,
        7.153604507446289,
        1.211376428604126
      ],
      "weighted_cancellation_ratio": 0.7092990875244141,
      "shared_history_to_expert_norm_ratio": 4.151325225830078,
      "physical_bound_fraction": 0.0
    }
  },
  "native_gradient_norms": {
    "commander": 0.04135255975593039,
    "region": 0.003314051936685061,
    "discriminate": 0.01097166764062243,
    "sensitivity": 0.0071566872435834174,
    "sequential": 0.08282881538923392,
    "coordinator": 0.014673436010477369,
    "fusion": 0.21727017567995385,
    "adapter2": 0.00368708274555975,
    "adapter4": 0.002418645522033849,
    "response_gate": 0.01037141019251746,
    "response_fusion": 0.020774690008995914
  },
  "new_router_head_native_gradient_norm": 0.0005943242943227663,
  "fixed_history_critic_not_EIG": true,
  "head_compensation_and_redundancy_not_causally_proved": true
}
```
