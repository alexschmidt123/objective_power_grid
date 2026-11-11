# Frozen evaluation protocol

One seven-configuration diagnostic,no training. Full HEC-v2,four one-hot expert pathways,uniform feature coordination,and compatible frozen conventional direct-head MoE. Same original independent-from-training HEC-v2 development bank:2000 trajectories,T30,1M negative prior contrasts plus each positive once. This reused diagnostic bank is not a new final test. Same truth and observation noise;policies are deterministic,legacy routing uniforms are preserved but not used to randomly select HEC experts.

Use the exact preserved full-horizon log-sum-exp contrast scorer,batch16,contrast_chunk16384. All changed-policy observations are resimulated. Literal tiny-estimator validation confirms denominator12 for11 negative contrasts plus positive. No contrast subsampling or policy optimization.

Primary five full-minus-ablation comparisons use paired t95%CIs(unadjusted intervals) and Holm5 adjusted p-values. Report the best observed individual expert and its already-corrected comparison;do not infer population ranking from a noisy winner. Full-minus-conventional is a secondary comparison with raw p and supplementary Holm6 across all six. Baseline absence/incompatibility is reported,not silently replaced.

GPU lease/duplicate-safe launch after source validation and LabPC->GitHub->Ubuntu sync. Queue if occupied;never interrupt jobs. Evaluation budget3600seconds excluding waiting;stop and report BLOCKED on correctness failure or timeout. Resource telemetry every2seconds;per-policy evaluation wall/CPU times,Torch peak GPU allocations,sampled device memory/utilization,process RAM,and100 single-case inference latency repetitions. Record physical simulator observations separately from negative/positive density terms and internal model queries. No additional training or architecture sweep.

Protect original checkpoint,source,bank,scores and artifact catalog hashes before/after. Save all outputs under the new moe_hec_v2_internal_ablation root. Source/configuration/evidence locked in PRE_EVALUATION_MANIFEST before evaluation. Generated reports contain actual values only and are synchronized after validation;no checkpoint/raw bank in Git.
