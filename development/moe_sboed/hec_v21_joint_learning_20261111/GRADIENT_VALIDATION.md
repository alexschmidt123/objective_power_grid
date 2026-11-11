# Correctness evidence

```json
{
  "GRADIENT_VALIDATION.json": {
    "status": "PASS",
    "seconds": 4.616353650810197,
    "scientific_optimizer_updates": 0,
    "initial_actions_bitwise_identical": true,
    "initial_inherited_parameter_gradients_match": true,
    "exact_collaborative_logit_action_tangent": true,
    "exact_design_likelihood_derivative": true,
    "new_native_router_head_gradient_positive": true,
    "full_gradient_paths": {
      "history": 0.4199350671026707,
      "commander": 0.1657871447008399,
      "coordinator": 0.038641343596226355,
      "response_gate": 0.029600297738304845,
      "region": 0.030372147579209207,
      "discriminate": 0.14148970144166562,
      "sensitivity": 0.03793276209387771,
      "sequential": 0.30826255108485834,
      "adapter2": 0.03644219630590272,
      "adapter4": 0.0115748622162644,
      "fusion": 2.2689539081456314,
      "response_fusion": 0.1873570632499272,
      "contribution_projection": 1.912513390865356e-05,
      "relevance": 0.0017360166332714899
    },
    "command_and_weight_action_effects": true,
    "prefix_future_mutation_invariant": true,
    "compatible_Adam_objects_and_moments_preserved": true,
    "full_native_batch": 10000,
    "actor_rows": 300000,
    "intercepted_steps": 32,
    "fixture_episodes": 4,
    "native_gamma1_reward_telescope": true,
    "checkpoints_unchanged": true,
    "peak_GPU_bytes": 2701801472
  },
  "EVALUATION_CHECKS.json": {
    "status": "PASS",
    "seconds": 18.413635747041553,
    "literal_contrast_convention": true,
    "repeat_full_trajectory_bitwise": true,
    "all_frozen_ablation_interfaces": true,
    "actor_state_roundtrip_bitwise": true,
    "competency_probe_pipeline_valid": true,
    "models_unchanged": true,
    "training_updates": 0
  },
  "CONDITIONAL_CHECKS.json": {
    "status": "PASS",
    "complete_conditional_resimulation_pipeline": true,
    "identical_continuation_consistency": true,
    "no_saved_future_replay": true,
    "grouped_native_return_not_exact_conditional_MI": true,
    "training_updates": 0
  }
}
```

Includes initial action and inherited gradient identity,exact gate-action/likelihood derivatives,all trainable pathway reachability,command/weight action effects,no future-prefix leakage,unchanged frozen checkpoint bytes,and full native10000-trajectory/300000-history batch with all32 fixture optimizer calls intercepted. No scientific optimizer steps. Evaluator/literal contrast,conditional resimulation,and external frozen loader receipts are separately required before training. No assertion is weakened.
