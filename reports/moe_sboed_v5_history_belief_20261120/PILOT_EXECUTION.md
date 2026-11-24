# Matched policy-only V5 pilot

One exploratory seed101 learned/uniform pair,200warmup+1800joint,original native critic/posterior/objective. Original canonical V2/V4/DAD/vsOED evaluation only. No new NN/parameters;zero initial action difference from previous V5.

Controller waits for the running GPU job. Then it runs one bounded two-update native correctness/timing preflight(discarded),learned training,uniform training sequentially. It stops on failure without automatic retraining or tuning. Every formal arm reloads original parent/seed. Both receive full matched updates and simulator episodes. Four-hour caps include conservatively reserved180sec preparation and evaluation. Actual preparation resources are separately measured;reserve is not actual runtime. No additional seeds.
