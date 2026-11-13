# Validation

Actual-checkpoint GPU smoke run completed with20 independent episodes and reduced teacher replicates. Representation assertions verify E1/E3 packets are preserved exactly while only E2/E4 analytic prefixes are excluded. A synthetic linear-target test checks ridge fitting and prediction on heldout data. Audit runs the original v2_tasks.labels, restores hash-verified V2, checks finite physical-action sensitivity, and asserts identical actor state plus historical/source hashes at completion. Zero optimizer steps.

Calibration, rival-selection and test slices are disjoint at episode granularity; calibration statistics/decoder weights never use rival selection or test data. Rival choice uses selection only. Smoke scores are correctness evidence only, not specialization results. Current-V2 teacher versus historical-V1 teacher is recorded as a limitation.
