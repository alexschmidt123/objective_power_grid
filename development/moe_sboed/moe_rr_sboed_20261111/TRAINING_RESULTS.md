# RR-MoE native training

Experiment status: COMPLETE.

One common 100-update uniform-Top2 preparation followed by 2000 learned and 2000 uniform joint updates. Each arm inherits the identical preparation actor/target,critic/posterior,replay,Adam/scheduler and simulator RNG. Frozen HEC is excluded from Adam and Polyak; no extra reward or training extension.

Actual preparation work is counted once (100000 fresh episodes); each arm inherits it and adds 2M episodes. Shared preparation is not a free pretrained model. The disposable native integration validation adds 1000 episodes/30000 observations and never initializes scientific training.

```json
{
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
  "full": {
    "status": "COMPLETE",
    "arm": "full",
    "updates": 2000,
    "seconds": 4384.294179705903,
    "process_CPU_seconds": 4359.5099177110005,
    "process_peak_RSS_bytes": 2697830400,
    "optimizer_steps": {
      "actor": 2000,
      "critic": 10000,
      "posterior": 300000
    },
    "simulator_counts": {
      "episodes": 2000000,
      "observations": 60000000,
      "batch_calls": 60000
    },
    "pair_selection_counts": [
      1619337,
      3663526,
      1710822,
      4387729,
      1733706,
      46884880
    ],
    "model_queries": {
      "frozen_belief_rows": 638000000,
      "frozen_SG_provisional_rows": 660000000,
      "exact_signal_evaluations": 411840000000,
      "analytic_parameter_jacobians": 42240000000,
      "frozen_opportunity_critic_rows": 1980000000
    },
    "GPU_peak_allocated_bytes": 3371904512,
    "actor_sha256": "e400e734ed20c4de2eeecad15b61448fd3f352bc40048ca9ee5633ce19e230b4",
    "runtime_sha256": "6cea3c7a2515ff1282f62abcc2423f6ad5b86f09d2294ac2ee5791de7267ed03",
    "frozen_base_unchanged": true,
    "trainable_parameters": 43972
  },
  "control": {
    "status": "COMPLETE",
    "arm": "control",
    "updates": 2000,
    "seconds": 4144.2237725860905,
    "process_CPU_seconds": 4115.245551292999,
    "process_peak_RSS_bytes": 2703306752,
    "optimizer_steps": {
      "actor": 2000,
      "critic": 10000,
      "posterior": 300000
    },
    "simulator_counts": {
      "episodes": 2000000,
      "observations": 60000000,
      "batch_calls": 60000
    },
    "pair_selection_counts": [
      9998851,
      10004754,
      9999896,
      9999539,
      9998569,
      9998391
    ],
    "model_queries": {
      "frozen_belief_rows": 638000000,
      "frozen_SG_provisional_rows": 660000000,
      "exact_signal_evaluations": 411840000000,
      "analytic_parameter_jacobians": 42240000000,
      "frozen_opportunity_critic_rows": 1980000000
    },
    "GPU_peak_allocated_bytes": 3371848704,
    "actor_sha256": "d071f0a5dbd9518ed055984e1da256aa4beccae1886c7e30be8ed221fe35df75",
    "runtime_sha256": "18209698886fda0ac6efb45adba74115f0ab6d95db252df8e44aad73dd7906cd",
    "frozen_base_unchanged": true,
    "trainable_parameters": 43972
  }
}
```
