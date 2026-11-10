# Protocol state

INCOMPLETE V3 PILOT — STOPPED AT THE PRE-TRAINING METHODOLOGICAL GATE

Preflight definition saved before frozen feature computation: seed 202611103001; stages0,3,12,24,29;256 distinct episodes/stage split128/128;upper-third calibration thresholds;positive weighting floor .1. No wins or policy outcomes tune these definitions. Source/inputs/evidence are hashed in PREFLIGHT_MANIFEST.json.

Training is NOT locked. CONFIG_LOCKED.json keeps its requested filename but explicitly says PREFLIGHT_ONLY_NOT_A_TRAINING_LOCK and training_launch_enabled=false. No PRETRAIN_MANIFEST, training launcher, background queue, optimizer preparation, final-test access, simulator training episode, or v3 checkpoint exists.

Unexecuted pilot outline only: dynamic-v3 versus same-architecture stage-only-v3;2000 native updates/arm as a starting reference;gamma1;original posterior/critic/Adam protocol where compatible;independent evaluation2000 trajectories/1M prior contrasts. No final preparation budget, learning rates, continuous action scales, exploration schedule, stochastic target MC count or runtime budget was selected. Inventing these as a locked protocol after a scientific no-go would be misleading.

The exact unresolved gate is a substantive, mathematically specified BOED-oriented joint responsibility mechanism distinct from the established stochastic mixture/native Q gradient and prior CIAS teacher. A theorem is not required, but an actual new learning rule and falsifiable rationale are. The inspected construction did not meet it.
