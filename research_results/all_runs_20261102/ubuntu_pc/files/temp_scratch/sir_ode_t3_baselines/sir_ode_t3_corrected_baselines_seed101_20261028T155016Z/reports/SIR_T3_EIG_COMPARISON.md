# SIR T3 EIG comparison

Terminal full-horizon explicit-likelihood sPCE,2048 paired confirmation trajectories; common L=1000000. One training seed, finite contrast bias/training-seed variance excluded from MC uncertainty.

| Method | EIG nats | MC SE |95%CI|
|---|---:|---:|---|
|random|4.221046|0.063777|[4.096043,4.346049]|
|fixed|2.708803|0.044891|[2.620816,2.796789]|
|rl_sboed|9.162122|0.032983|[9.097476,9.226769]|
|vsoed|9.548669|0.032464|[9.485040,9.612298]|

Readiness-blocked methods are absent, not assigned invented EIG. See paired_differences.json for all paired differences.
