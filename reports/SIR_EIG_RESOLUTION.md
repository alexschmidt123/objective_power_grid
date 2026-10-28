# SIR explicit-likelihood EIG resolution

Historical audits found substantial16,384→1,000,000 shifts for Fixed/DAD (about.58/.79nats), and existing1M→4M evidence did not certify the declared.02 absolute/.01 paired resolution tolerances. Therefore16,384 is not silently retained.

One independent calibration was performed on seed97402621,256 paired cases per Random/corrected historical DAD, with nested16,384/65,536/262,144/1,000,000 contrasts. No official-seed model tuning or method-ranking criterion was used. The common primary evaluation count is frozen at1,000,000 for all six methods; residual finite-contrast bias is explicitly unresolved rather than represented as negligible. MC standard error describes trajectory Monte Carlo variability, not estimator bias or training-seed variability. Reserved confirmation randomness is unused.

| Calibration method | Contrasts | EIG | Shift to1M | Paired shift SE |
|---|---:|---:|---:|---:|
| random | 16384 | 4.241909 | -0.088073 | 0.024365 |
| random | 65536 | 4.306030 | -0.023952 | 0.011316 |
| random | 262144 | 4.329593 | -0.000389 | 0.006274 |
| random | 1000000 | 4.329982 | 0.000000 | 0.000000 |
| dad | 16384 | 4.034518 | -0.050058 | 0.021369 |
| dad | 65536 | 4.057361 | -0.027214 | 0.015626 |
| dad | 262144 | 4.079943 | -0.004633 | 0.008062 |
| dad | 1000000 | 4.084576 | 0.000000 | 0.000000 |
