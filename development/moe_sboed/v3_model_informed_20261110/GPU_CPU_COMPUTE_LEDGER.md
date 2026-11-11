# Actual compute

```json
{
  "preparation": null,
  "arms": {
    "stage": null,
    "candidate": null
  },
  "evaluation": null,
  "latency": null,
  "resources": {},
  "total_seconds": null
}
```

Raw10second CPU,RAM,GPU utilization/memory/power samples:TRAINING_LOGS/resources.jsonl. CPU percent is relative to one core;child cumulative CPU is separately logged. Device samples are not exact per-kernel accounting or energy integration. Preparation is physically shared once and charged equally for training opportunities. Frozen provider costs/buffers are included;no sparse-FLOP claim or EIG-per-compute comparison across unmatched frozen baselines.
