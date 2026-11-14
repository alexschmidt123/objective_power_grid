## Mandatory MoE-sBOED evaluation comparability — user instruction 2026-11-14

Primary comparisons must use the exact saved HEC-v2 evaluation for V2, DAD, vsOED and other baselines. Never silently substitute banks, seeds, case counts, contrast counts, scorers, physics or checkpoint variants.

- Canonical bank: TEMP/scratch_runs/moe_hec_v2_icml_20261109/evaluation/development_validation_bank.npz;2000 truth/noise cases and1000000 prior contrasts, byte-identical to V2. Routing uniforms come from the same original evaluation directory.
- Use V2's frozen float64 likelihood scorer, chunk16384, positive truth included once, full T=30, sigma0.5, two normally distributed2D sources, and2D design bounds[-4,4]. Preserve each baseline's validated inference implementation and evaluation-mode behavior.
- Reproduce the entire saved V2 learned/uniform, DAD and vsOED score vectors within1e-9 before accepting comparison results. Also reproduce RL-sBOED and ScoreBED when included. Verify bank, scorer, checkpoints and saved-reference hashes; report paired uncertainty and multiplicity correction.
- Include the current learned candidate, original V3, its uniform and all four standalone experts, V2 learned/uniform, DAD, vsOED and the other established fixed baseline checkpoints. Missing baselines are blockers or explicitly incomplete comparisons, never silent exclusions.
- Different contrast counts or banks are supplementary only; never replace this primary benchmark or compare their absolute EIG values to historical scores. Changing the primary protocol requires explicit user instruction. Do not select checkpoints or tune hyperparameters on the reference evaluation.
- Evaluation matching does not imply training-budget matching. State training provenance and budget differences separately. Historical baselines remain frozen references, not newly promoted official results.
- Preserve all historical artifacts. New evaluation outputs go to a new TEMP/scratch_runs directory. No model training is authorized by this rule itself.

The executable guard and rerun entry point are in development/moe_sboed/hec_canonical_evaluation_20261114/. This rule applies to future MoE-sBOED implementation and result reporting.
