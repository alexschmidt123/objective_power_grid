# Validation

Actual-checkpoint GPU smoke run completed with20 independent episodes and reduced teacher replicates. Representation assertions verify E1/E3 packets are preserved exactly while only E2/E4 analytic prefixes are excluded. A synthetic linear-target test checks ridge fitting and prediction on heldout data. Audit runs the original v2_tasks.labels, restores hash-verified V2, checks finite physical-action sensitivity, and asserts identical actor state plus historical/source hashes at completion. Zero optimizer steps.

Calibration, rival-selection and test slices are disjoint at episode granularity; calibration statistics/decoder weights never use rival selection or test data. Rival choice uses selection only. Smoke scores are correctness evidence only, not specialization results. Current-V2 teacher versus historical-V1 teacher is recorded as a limitation.

The full640-episode audit completed with unchanged actor/artifact hashes. A second independent16-replicate teacher pass estimates MC target noise before selecting an intervention. E4 residual above that noise is0.159596,95%CI[0.114027,0.205165].

Actual-checkpoint refresh smoke test: three preparation optimizer steps for each arm. Control head-only refresh changes actions by exactly0; candidate adapter+head refresh changes actions (maximum0.000495315). All other actor parameters remain bitwise identical, gradients finite, native optimizer moments untouched, native training steps0. Full experiment uses300 fixed preparation steps and separately locked2000 native updates per arm; the smoke test does not establish improved competency or EIG.
