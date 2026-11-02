# Corrected PIMoE-IC training

Source commit: 354021f4d9abda9e4dd7ed721c2edbd97c716460. Seed101, fresh initialization, RTX4090. Architecture unchanged: four independent10-48-48-9 tanh density heads with three Gaussian components each,11-32-4 router;13800 parameters. No invalid checkpoint weights reused.

Numerical correction uses log-domain posterior-region probabilities and scale-invariant conditional ESS.23 tests passed, including reproduction of the invalid oldESS, rare-region gradients, density normalization, checkpoint rejection and source-label symmetry.

Completed14000 updates in70.521546s, final checkpoint SHA256 3f27a60327fe18a34cddd011eed6caad57cde67b03f9fa32ae279093092c14c1. Adam.001/globalclip5; unchanged information, ranking and conditional-likelihood coefficients. Latest valid update, no official-EIG selection.

Prior invalid optimizer cost70.630124382s remains counted. Cumulative optimizer cost141.151670s EXCEEDS the prior cumulative70.918615084s cap. Corrected attempt alone is within70.918615084s. Cumulative measured experimental runtime197.338507s remains below1210.631910391s. Exact equal end-to-end computation is NOT established: original record included development reservations, and GPU-active time is unmeasured. Official evaluation is separate. No budget reset.

Data: six complete exploratory training episodes,24belief states, independent training observation and reference-target streams; three held-out non-neural-greedy episodes/12states, seeds1010/1011. Existing source simulator, prior, likelihood, bounds andT30 unchanged. Posterior inference remains approximate rejuvenated particles. Reference-target uncertainty saved per candidate.
