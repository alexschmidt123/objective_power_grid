# HEC-v2 counterfactual routing pilot result

This experiment keeps the actual successful V2 history, commander, four experts, approximate belief, SG reference and fusion frozen. It changes only coordination: a small value-selector chooses original, uniform or one of six conservative pair-favoring gate patterns. It learns paired terminal information advantages from direct simulated counterfactuals, without critic targets. It remains dense V2 computation; it is not sparse execution or RR-MoE.

Outcome: **ROUTING SUPERIORITY NOT ESTABLISHED**. Completed in 117.10s. Frozen V2 and all protected historical artifacts unchanged.

## Fresh full-trajectory evaluation

512 paired complete T30 trajectories;1M independent prior negatives plus the positive once. Fresh final bank was never used for fitting, model/threshold selection or checkpoint selection. Scores are finite-contrast lower-bound estimates; uncertainty is Monte Carlo conditional on a single frozen lineage and bank, not training-seed variability. All seven candidate contrasts use one Holm family; CIs are unadjusted95% paired t intervals. One-hot controls retain the shared history/head/SG reference; they are not separately trained standalone expert policies. External conventional baselines were not reevaluated in this pilot.

| Policy | EIG nats | MCSE |
|---|---:|---:|
|candidate|12.810432|0.072340|
|original|12.901423|0.074087|
|uniform|12.883940|0.070018|
|stage|12.712903|0.075361|
|expert1|11.153802|0.099528|
|expert2|11.765660|0.100899|
|expert3|11.257099|0.107818|
|expert4|12.406388|0.086744|

| Candidate minus | Difference nats | 95% paired CI | Holm p |
|---|---:|---|---:|
|original|-0.090990|[-0.197462,+0.015481]|0.187544|
|uniform|-0.073508|[-0.175359,+0.028344]|0.187544|
|stage|+0.097530|[-0.000618,+0.195678]|0.154367|
|expert1|+1.656631|[+1.457389,+1.855872]|1.00642e-47|
|expert2|+1.044773|[+0.855451,+1.234094]|4.30304e-24|
|expert3|+1.553333|[+1.342025,+1.764642]|4.09218e-39|
|expert4|+0.404044|[+0.242584,+0.565504]|4.76017e-06|

## Interpretation and limitations

Success requires a positive Holm-significant difference against original V2, uniform V2, the stage-only selector AND all four one-hot controls. A higher point estimate alone is insufficient. If this criterion fails, the candidate is not promoted as an improved scientific model and no follow-up campaign is launched automatically.

Teacher labels use one random intervention per independently generated original-V2 episode and one true latent/future-noise realization per history. Alternatives share that realization and contrast bank. Latent variables and future noise enter scoring only, never routing inputs. The selector is fitted with exact ridge regression from512 observed-history contexts;128 separate teacher contexts are diagnostic only. This is noisy policy-improvement supervision. Repeated new routing at every decision shifts states and continuation away from the one-intervention frozen-V2 teacher; this mismatch and restricted gate bank are explicit limitations, not proven explanations for the outcome. There is no guaranteed improvement from using direct returns.

## Execution and preservation

Five CPU unit tests and a complete GPU integration smoke passed. The first scientific attempt stopped before any router fitting because recomputing the unchanged reference in a larger float32 GEMM batch caused history-accumulated differences up to3.618e-5. It is preserved at `scratch_runs/hec_v2_counterfactual_routing_20261112_pilot/`. The repair reuses the exact saved original rollout for that branch and asserts bitwise equality of actions and observations. The repaired run kept the same seeds, sizes and budget; no performance-selected rerun occurred.

Source commit: `b8b27a21a0653290cbb62545c25cf88d90f7561e`.

Scientific output: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v2_counterfactual_routing_20261112_repaired_pilot`.

Original actor SHA256: `90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108`.

Original checkpoint weights and all protected source/runtime files passed preservation checks. New source was synchronized LabPC→GitHub→Ubuntu before launch; computation occurred only on LabPC. Raw teachers, learned router checkpoint and score arrays stay outside Git. SUMMARY.json contains compact receipts and hashes.
