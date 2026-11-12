# Actual compute ledger

Experiment status: NOT_STARTED_OR_RUNNING.

Cost is recorded only as a secondary measure,never an optimization term. Training receipts give actual optimizer/simulator calls,wall time,CPU time,peak RSS and CUDA allocations. Frozen-cache construction and internal HEC likelihood/Jacobian/provider queries are counted by model counters. Cache reuse preserves the same frozen inputs and values.

Device utilization/power/memory and process CPU/RSS are sampled every10seconds. The full frozen HEC model still executes; RR does not claim two-expert cost for the entire policy. Inference latency includes frozen HEC providers and two selected new adaptations. Shared preparation costs are separate,not omitted from either lineage. Evaluation receipts count prior-likelihood terms and observations.

```json
{
  "stages": {
    "preparation": null,
    "full": null,
    "control": null
  },
  "resources": {},
  "latency": null,
  "evaluation_completion": null
}
```
