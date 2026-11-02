# Posterior diagnostic findings
Actual location-finding T30 likelihood, fixed frozen V4/V5 histories, no optimizer update. Six prefixes (0,15,29), three independently initialized resample-move populations for each512/8 versus4096/32 setting. Reference is a higher-resolution approximation, NOT an exact oracle. Neither group is tuned from this diagnostic.

|Policy|Prefix|Executed conditional score|Reference score|Executed−reference ± replica MC SE|Maximum probe predictive-mean discrepancy|
|---|---:|---:|---:|---:|---:|
|V5A|0|0.09761|0.08259|0.01502 ± 0.01909|0.02709|
|V5A|15|0.26259|0.24253|0.02006 ± 0.01733|0.00808|
|V5A|29|-0.15385|-0.15292|-0.00092 ± 0.00539|0.00364|
|V4|0|0.77347|0.85061|-0.07714 ± 0.01351|0.02709|
|V4|15|0.01393|0.01289|0.00104 ± 0.00091|0.00838|
|V4|29|0.01611|0.02376|-0.00765 ± 0.00437|0.00406|

Each population score averages16 random128-negative subsets; replica SE uses only3 populations, and paired replica seeds do not make pool errors independent. These are fixed realized information scores, not conditional expected EIG. At prefix0 the two settings both sample the exact prior; their finite-pool differences are Monte Carlo error, not evidence of inference bias. Later differences indicate sensitivity, not a measured exact-posterior bias.

Source audit: iid standard-normal four-coordinate prior; Gaussian log-signal observation withsigma0.5. All prefix likelihoods included. Adaptive tempering enforces pre-resampling ESS; MH targets log prior+old log likelihood+tempered next log likelihood. Symmetric source swaps preserve likelihood exactly on every tested pool. Swaps alone do NOT demonstrate physical mode coverage. Final ESS is uniform by construction and does not certify quality. Finite populations/locally adapted MH may miss modes. Proposal adaptation is finite and nonstationary, not an exact posterior certificate.

Prefix/current/future-noise perturbation passed: actual production call supplies exactly15 past actions/observations, no truth or current/future observation. Same prefix produced byte-identical negatives despite changed future noise. See prefix_and_surrogate_checks.json.

Acceptance, unique fractions, pre-resampling ESS, label symmetry, predictive CDF and particle-order correlations are inposterior_diagnostics.csv. Particle-order correlation is NOT chain autocorrelation; MH traces and integrated autocorrelation times are UNAVAILABLE. Three replicas on six prefixes do not establish calibration/multimodal coverage over the whole learned policy distribution. Earlier128-history calibration passed but used V4 histories and broad thresholds. It does not certify the evolving V5 policy's posterior.

The deployed policy uses only history. OFFLINE inference is detached; gradient through the history-conditioned posterior distribution is omitted. Actual stopped-pool surrogate derivative agrees with finite differences (absolute error1.49e-8), which verifies the implementation of that surrogate, not the gradient of the intended conditional-EIG functional. No rigorous lower-bound/unbiased-gradient claim is justified.
