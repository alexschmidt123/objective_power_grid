# Actual compute ledger

No training resource claim is made. CPU/GPU utilization and energy time series were not sampled;do not infer full utilization from wall time. ExistingGPUjobs were checked before the frozen audit;RTX4090 idle except display processes. The inherited nonblockingGPUleases were acquired;no jobs were interrupted.

```json
{
  "gradient_fixture_seconds": 0.21248311316594481,
  "frozen_audit_wall_seconds": 3.7884274960961193,
  "GPU": "NVIDIA GeForce RTX 4090",
  "peak_GPU_allocated_bytes": 1449875968,
  "training_updates": 0,
  "preparation_updates": 0,
  "new_simulator_episodes": 0,
  "new_simulated_observations": 0,
  "model_query_counts": {
    "frozen_belief_rows": 2048,
    "frozen_SG_provisional_rows": 1280,
    "exact_signal_evaluations": 880642,
    "analytic_parameter_jacobians": 1196034,
    "frozen_opportunity_critic_rows": 3840
  },
  "CPU_utilization": "NOT_SAMPLED",
  "CPU_RSS": "NOT_SAMPLED",
  "GPU_utilization_time_series": "NOT_SAMPLED",
  "new_EIG_evaluation_seconds": 0,
  "v3_inference_latency": "NOT_AVAILABLE_NO_MODEL"
}
```

The frozen wall time includes loading,feature computation and hashing of protected inputs. These timings are not EIG per unit of training compute. Network/research/documentation time is not a training measurement. A future pilot would need explicitGPU,CPU,RAM telemetry, simulator and exact-likelihood counts, parameter counts,policy latency and separate preparation/training/evaluation clocks.
