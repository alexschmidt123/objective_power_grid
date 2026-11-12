# RR-MoE native training

Experiment status: NOT_STARTED_OR_RUNNING.

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
  "full": null,
  "control": null
}
```
