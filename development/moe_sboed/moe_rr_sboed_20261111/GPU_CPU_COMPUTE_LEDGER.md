# Actual compute ledger

Experiment status: NOT_STARTED_OR_RUNNING.

Cost is recorded only as a secondary measure,never an optimization term. Training receipts give actual optimizer/simulator calls,wall time,CPU time,peak RSS and CUDA allocations. Frozen-cache construction and internal HEC likelihood/Jacobian/provider queries are counted by model counters. Cache reuse preserves the same frozen inputs and values.

Device utilization/power/memory and process CPU/RSS are sampled every10seconds. The full frozen HEC model still executes; RR does not claim two-expert cost for the entire policy. Inference latency includes frozen HEC providers and two selected new adaptations. Shared preparation costs are separate,not omitted from either lineage. Evaluation receipts count prior-likelihood terms and observations.

```json
{
  "stages": {
    "preparation": {
      "status": "COMPLETE",
      "arm": "preparation",
      "updates": 100,
      "seconds": 206.7178191859275,
      "process_CPU_seconds": 204.58725667,
      "process_peak_RSS_bytes": 2681769984,
      "optimizer_steps": {
        "actor": 100,
        "critic": 500,
        "posterior": 15000
      },
      "simulator_counts": {
        "episodes": 100000,
        "observations": 3000000,
        "batch_calls": 3000
      },
      "pair_selection_counts": [
        499942,
        499896,
        500603,
        500605,
        499253,
        499701
      ],
      "model_queries": {
        "frozen_belief_rows": 31900000,
        "frozen_SG_provisional_rows": 33000000,
        "exact_signal_evaluations": 20592000000,
        "analytic_parameter_jacobians": 2112000000,
        "frozen_opportunity_critic_rows": 99000000
      },
      "GPU_peak_allocated_bytes": 3372894720,
      "actor_sha256": "45c7fd8c579c9ccc0b4e8d51431bf11cd281682be0566461f4a10c4e7b98ae7a",
      "runtime_sha256": "5b318e06d8830b3a8efdfb8a9e4627acad76c0898106e3211d57e427f88daee4",
      "frozen_base_unchanged": true,
      "trainable_parameters": 43972
    },
    "full": null,
    "control": null
  },
  "resources": {
    "CPU_percent_one_core": {
      "mean": 99.24325404468735,
      "maximum": 100.07115097473604
    },
    "RSS_bytes": {
      "mean": 2106589801.2054794,
      "maximum": 2202279936.0
    },
    "GPU_percent": {
      "mean": 57.84931506849315,
      "maximum": 72.0
    },
    "GPU_memory_MiB": {
      "mean": 4627.054794520548,
      "maximum": 4735.0
    },
    "GPU_power_W": {
      "mean": 158.76671232876714,
      "maximum": 183.4
    }
  },
  "latency": null,
  "evaluation_completion": null
}
```
