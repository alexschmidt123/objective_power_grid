# Actual compute ledger

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
  "evaluation": {
    "status": "COMPLETE",
    "seconds": 574.2723484109156,
    "policies_completed": [
      "original",
      "conventional",
      "vsoed",
      "continuation",
      "candidate",
      "uniform",
      "expert1",
      "expert2",
      "expert3",
      "expert4",
      "dad",
      "rl_sboed",
      "scorebed"
    ],
    "no_training_updates": true,
    "checkpoints_unchanged": true
  },
  "latency": {
    "milliseconds_per_design": {
      "candidate": 3.6553658498451114,
      "continuation": 2.9716401593759656,
      "original": 2.969814690295607,
      "conventional": 0.1661442592740059
    },
    "history_stage": 12,
    "repeats": 100,
    "includes_all_frozen_provider_queries": true
  },
  "sampled_resources": {
    "CPU_percent_one_core": {
      "sample_mean": 98.62461553799979,
      "sample_max": 100.06263711377741
    },
    "RSS_bytes": {
      "sample_mean": 2108459763.5179393,
      "sample_max": 2778521600.0
    },
    "GPU_percent": {
      "sample_mean": 65.84084636614536,
      "sample_max": 100.0
    },
    "GPU_memory_MiB": {
      "sample_mean": 3749.7470101195954,
      "sample_max": 3925.0
    },
    "GPU_power_W": {
      "sample_mean": 181.2083716651334,
      "sample_max": 219.15
    },
    "samples": 1087,
    "interval_seconds": 10
  },
  "total_driver_seconds": 10902.266097717918
}
```

CPU percentages are relative to one core;child cumulative CPU is separately logged. RAM samples and process high-water RSS,deviceGPU memory samples and Torch allocated peaks are different measurements. Resource samples span training/evaluation;per-arm times separate training. Queries include analytic feature computations and checkpoint recomputation,not only physical simulation. No cost enters the objective and no unmatched EIG-per-compute superiority is inferred.
