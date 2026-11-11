# Actual compute ledger

```json
{
  "arms": {
    "continuation": null,
    "candidate": null
  },
  "evaluation": null,
  "latency": null,
  "sampled_resources": {},
  "total_driver_seconds": null
}
```

CPU percentages are relative to one core;child cumulative CPU is separately logged. RAM samples and process high-water RSS,deviceGPU memory samples and Torch allocated peaks are different measurements. Resource samples span training/evaluation;per-arm times separate training. Queries include analytic feature computations and checkpoint recomputation,not only physical simulation. No cost enters the objective and no unmatched EIG-per-compute superiority is inferred.
