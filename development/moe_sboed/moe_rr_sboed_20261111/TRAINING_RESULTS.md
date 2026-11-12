# RR-MoE native training

Experiment status: NOT_STARTED_OR_RUNNING.

One common 100-update uniform-Top2 preparation followed by 2000 learned and 2000 uniform joint updates. Each arm inherits the identical preparation actor/target,critic/posterior,replay,Adam/scheduler and simulator RNG. Frozen HEC is excluded from Adam and Polyak; no extra reward or training extension.

Actual preparation work is counted once (100000 fresh episodes); each arm inherits it and adds 2M episodes. Shared preparation is not a free pretrained model. The disposable native integration validation adds 1000 episodes/30000 observations and never initializes scientific training.

```json
{
  "preparation": null,
  "full": null,
  "control": null
}
```
