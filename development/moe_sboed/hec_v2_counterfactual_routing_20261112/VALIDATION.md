# Validation checkpoint — 2026-11-12

LabPC dellma PyTorch environment. `PYTHONDONTWRITEBYTECODE=1`, CPU-only unit tests:

- Four tests passed: positive/simplex gate bank and literal first+response fusion; full T30 positive-once prior-sPCE vs literal density calculation and chunk invariance; held-out synthetic history-dependent coordination and train-only scaling; Holm family correction.
- Initial iterative fitting did not converge adequately on the synthetic held-out signal. Before any scientific run it was replaced with an exact, regularized linear least-squares solve. This removes optimizer convergence as a confound; it does not guarantee generalization on the actual noisy experiment.
- Integration smoke `scratch_runs/hec_v2_counterfactual_smoke_20261112a/`: SMOKE_COMPLETE,5.677 scientific wall seconds, all eight full-rollout policies completed. Original actor/runtime/source hashes preserved; run source unchanged; frozen V2 weights unchanged.
- Smoke31-negative scores saturate at log32 and are intentionally uninformative for scientific comparison. The actual pilot retains1M negatives, including for training labels. No smoke score is scientific evidence of improved routing.
- Read-only frozen baseline loading verifies exact state equality between the stored actor and runtime. The original branch in every counterfactual teacher fixture must reproduce the original rollout within documented floating-point tolerances. New source and frozen historical inputs are hashed before/after each run. No historical training/provider checkpoint is written.

Scientific run not completed at this source checkpoint. See the isolated run's COMPLETION.json and RESULTS.json for actual evidence. No scientific superiority is claimed here.

## Numerical reference-branch correction

Initial scientific launch `hec_v2_counterfactual_routing_20261112_pilot` stopped before fitting at training batch176. The unchanged reference was recomputed in a larger GEMM batch and float32 differences accumulated through history: maximum physical-design difference3.618e-5, initially1.192e-7. The run's preservation checks passed. No validation/final scientific result or router checkpoint existed. Its output is retained.

Corrected teacher reuses the original complete reference rollout exactly, omits its redundant resimulation, and checks candidate-zero action agreement at the actual intervention prefix. It now asserts bitwise equality of the saved reference branch's actions AND observations. A new CPU teacher integration unit test covers this path. This repairs numerical bookkeeping rather than loosening a tolerance or selecting a model based on results. The repaired launch uses the same seeds, teacher/evaluation sizes and wall cap in a new output directory.
