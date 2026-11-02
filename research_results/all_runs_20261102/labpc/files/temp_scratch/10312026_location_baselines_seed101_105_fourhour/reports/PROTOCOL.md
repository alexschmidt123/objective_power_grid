# Corrected location-finding five-seed campaign

T=30; two independent standard-normal 2D sources; log(.1 + sum 1/(1e-4+distance^2)); Gaussian observation sigma=.5; common [-4,4]^2 designs. Random retains truncated standard-normal designs.

Fresh initialization for seeds101–105. Native DAD full-horizon sPCE, native SAC/REDQ RL, native variational actor-critic-posterior vsOED, native two-stage ScoreBED; no shared episode quota. Fixed is one nonadaptive schedule optimized by joint sPCE.

Whole-method ceiling: 14400 seconds per seed/method, including preparation, pretraining, restarts and serialization. Native schedule completion can occur sooner. Budget exhaustion is COMPUTE_LIMITED, never convergence. No arbitrary tiny update target.

DAD:50000 updates B2000,L2000. Fixed:50000 updates B64,L1024,matching the completed corrected seed101 recipe; this is a disclosed method-specific offline optimizer setting, not equal updates forced across all algorithms. RL:20001 collection cycles B100,L100000 and64 native SAC/REDQ updates per eligible cycle. vsOED:10001 native cycles plus20000 initialization. ScoreBED:P1 reference190000 score updates and10000 policy updates, native B1024,one policy arm per training seed; the hardware-timing-only four-hour adaptation freezes177365 score updates then9335 policy updates, preserves native scheduler horizons, and is COMPUTE_LIMITED. No expensive stage is uncharged or silently dropped. This single-restart adaptation is explicit and is not an all-restarts paper reproduction.

Preserve common charged-time checkpoints at approximately0.5h,1h,2h,4h and thereafter every0.5h. Fair-compute comparisons and training-sufficiency comparisons are separate. Latest valid checkpoint, no final-EIG selection.

Independent development validation:128 cases,L1M,seed73831032, only to report progression. Final2000 paired cases,L1M,seed73831031, generated only after checkpoints are frozen. Raw paired scores, within-policy MCSE, between-training-seed SD and mean SE reported separately.

Training adequacy is not inferred from a time limit: report native completion, saturation, recent development improvements and their paired uncertainty. An evidently improving capped run is not an adequately trained scientific endpoint. No automatic extension, retuning, STABLE promotion or GitHub push.

Persistent controller never restarts a partial native phase from zero. Completed phases are reusable; partial phases remain blocked pending exact-state recovery.
