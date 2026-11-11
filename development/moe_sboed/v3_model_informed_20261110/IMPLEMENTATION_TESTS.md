# Correctness evidence

```json
{
  "IMPLEMENTATION_CHECKS.json": {
    "status": "PASS",
    "seconds": 8.534062440972775,
    "scientific_optimizer_steps": 0,
    "fixture_episodes": 4,
    "fixture_actual_simulator_counts": {
      "episodes": 4,
      "observations": 120,
      "batch_calls": 30
    },
    "intercepted_optimizer_calls": 32,
    "full_native_batch": 10000,
    "actor_history_rows": 300000,
    "all_30_stages_feasible_all_4_experts": true,
    "stage_gate_bitwise_history_invariant": true,
    "model_informed_router_full_chain_gradient_identity": true,
    "model_feature_likelihood_and_Jacobian_checks": true,
    "no_future_prefix_leak": true,
    "checkpoint_RNG_repeat_bitwise": true,
    "full_horizon_native_telescope": true,
    "compatible_encoder_Adam_preserved": true,
    "gradient_norms": {
      "encoder": 0.027948877273799577,
      "experts": 0.8659318557239759,
      "router": 0.009583577799499431
    },
    "peak_GPU_bytes": 2400620032,
    "transformed_native_train_sha256": "53b12478ce999ed9f5349a623da2683d6cea73fd584be4b7bc2b77d34528bd87"
  },
  "EVALUATION_CHECKS.json": {
    "status": "PASS",
    "seconds": 5.146306726848707,
    "literal_estimator_matches": true,
    "positive_included_once": true,
    "repeat_new_policy_rollouts_bitwise": true,
    "changed_policy_observations_resimulated": true,
    "intervention_interfaces_valid": true,
    "baseline_loaders": [
      "hec_v2",
      "hec_v1",
      "sg",
      "option",
      "collaborative",
      "ordinary",
      "vsoed"
    ],
    "checkpoints_unchanged": true,
    "scientific_training_updates": 0
  },
  "CONDITIONAL_CHECKS.json": {
    "status": "PASS",
    "identical_policies_exact_zero": true,
    "changed_policy_suffix_resimulation_finite": true,
    "zero_variance_interaction_handled": true,
    "training_updates": 0,
    "seconds": 6.125710000982508
  }
}
```

Additional evidence:CONDITION_PREFLIGHT.json,EVALUATION_CHECKS.json,CONDITIONAL_CHECKS.json and three historical-baseline preflights. The first dynamic-variation fixture compared two identical prior histories;it was corrected to distinct same-stage observed histories without weakening the assertion. Failed logs are preserved. No optimizer stepped in correctness fixtures.
