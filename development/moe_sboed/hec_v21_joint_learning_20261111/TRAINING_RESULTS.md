# Actual native training

```json
{
  "arms": {
    "continuation": {
      "status": "COMPLETE",
      "arm": "continuation",
      "updates": 2000,
      "seconds": 4732.195917099947,
      "seconds_per_update": 2.3660979591564972,
      "process_CPU_seconds": 4710.6265425110005,
      "process_max_RSS_bytes": 2749673472,
      "optimizer_steps": {
        "actor": 2000,
        "critic": 10000,
        "posterior": 300000
      },
      "actual_simulator_counts": {
        "episodes": 2000000,
        "observations": 60000000,
        "batch_calls": 60000
      },
      "model_queries": {
        "frozen_belief_rows": 1218002624,
        "frozen_SG_provisional_rows": 1240002624,
        "exact_signal_evaluations": 1148161637376,
        "analytic_parameter_jacobians": 117760167936,
        "frozen_opportunity_critic_rows": 5520007872
      },
      "parameters": 189822,
      "frozen_buffer_elements": 15266041,
      "GPU_peak_allocated_bytes": 2691441664,
      "actor_sha256": "876d1cce5a8a1cbf0edb48b81a71ad27cc96cd09a60010ccfab2c5a4d5c2dda4",
      "runtime_sha256": "998684c0afdf9beaeb6bbd4092217174fa72638039f89716b6207e0e6c20344a",
      "all_frozen_buffers_unchanged": true
    },
    "candidate": {
      "status": "COMPLETE",
      "arm": "candidate",
      "updates": 2000,
      "seconds": 5592.119116814109,
      "seconds_per_update": 2.796059559185989,
      "process_CPU_seconds": 5569.871255952,
      "process_max_RSS_bytes": 2776690688,
      "optimizer_steps": {
        "actor": 2000,
        "critic": 10000,
        "posterior": 300000
      },
      "actual_simulator_counts": {
        "episodes": 2000000,
        "observations": 60000000,
        "batch_calls": 60000
      },
      "model_queries": {
        "frozen_belief_rows": 1218002624,
        "frozen_SG_provisional_rows": 1240002624,
        "exact_signal_evaluations": 1148161637376,
        "analytic_parameter_jacobians": 235520335872,
        "frozen_opportunity_critic_rows": 5520007872
      },
      "parameters": 203839,
      "frozen_buffer_elements": 15266041,
      "GPU_peak_allocated_bytes": 2691496448,
      "actor_sha256": "23db0e82105a46b7df5200224c3c6eab10bf177c419b8f439da34bda33bfe17e",
      "runtime_sha256": "6077b77f3e8763709d199a431d25a17905dda745df5ab3b61001ff61f88ab7e7",
      "all_frozen_buffers_unchanged": true
    }
  },
  "completion": {
    "status": "COMPLETE",
    "seconds": 10902.266097717918,
    "no_additional_training": true
  }
}
```

No auxiliary objective or preparation. Original checkpoints,optimizer/replay lineage and all frozen anchors are preserved. Per-arm counts are measured from actual simulator and optimizer calls. Partial runs remain incomplete;no budget extension.
