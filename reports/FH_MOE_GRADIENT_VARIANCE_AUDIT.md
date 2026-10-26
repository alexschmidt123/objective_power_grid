# FH-MoE full-horizon gradient signal audit

Outcome: v3 reproduces; substantial finite-contrast estimation noise exists, but the predeclared multi-stage context criterion is not met after tested variance reduction. No policy was trained and no official EIG evaluation was run.

## Provenance and original calculations

Base commit5926e50389bc99b6fc98d2742ca2fe37226482ba. All16 preserved v3 artifacts were SHA256 verified, including three8000-case gradient datasets and the frozen preparation policy. The original stage1/10/20/29 metrics and all acceptance requirements remain unchanged. Reproduction uses the saved policy/data, not a preparation rerun. The new plan was written before GPU collection; SHA2569db8756767607f22318eb679510e26818ea1c63e759d17598821b8527081ce23.

For unit-label vectors g, historical coherence is ||mean(g)||. SNR is ||mean(g)|| / sqrt(trace(sample_cov(g))/n), calculated independently in each test context and even/odd split. Both contexts and both splits must meet coherence>=.1,SNR>=3. Other original gates include held-out MSE improvement, support>=10%,split-direction cosine>=.8,between-context cosine<.8,kappa>=.6,and numerical agreement>=.99. SNR is a heuristic norm statistic with positive null bias, not a calibrated probability or a t-test. Historical MSE uses mean+/-1.96case-levelSE; original kappa/coherence/cosines had no population CI. Added fixed-label bootstrap200 intervals condition on fitted groups and do not include refitting uncertainty. Independent refitting supplies a separate replication diagnostic. None is familywise confirmation.

## A, B, H: raw norms, normalization and directional concentration

| Stage | Raw norm median | Raw norm p99 | Fraction <=1e-12 | Raw cancellation ratio | Whole-split unit mean norm |
|---|---:|---:|---:|---:|---:|
|1|0.103025|5.23539|1.850%|0.067610|0.028932|
|10|0.0244157|1.15626|2.125%|0.050103|0.015786|
|20|0.0198954|1.77504|2.212%|0.084518|0.047948|
|29|0.00851711|0.980013|2.450%|0.146733|0.074697|

Raw norms span more than50orders of magnitude in these saved cases. Normalizing before averaging gives almost-vanishing gradients comparable directional weight to large gradients when their norms exceed the1e-30normalization floor. Below that floor vectors have norm<1; they are not mathematically unit vectors. Norm quantiles and covariance matrices are exact descriptions of these saved samples, not uncertainty-free population estimates.

Descriptively dropping the bottom5%of raw norms does not consistently rescue contexts: stage10 context coherences become.07463/.07244,stage20.05721/.08520,and stage29.06616/.13806. The original gate was not recalculated with lower thresholds or adopted trimmed data. Tiny-gradient amplification contributes to a changed target, but is not established as the sole cause. Finite-sPCE saturation is a plausible mechanism for very small gradients; no exact per-case saturation attribution was measured.

## C, D, E, I: covariance, separation, independent replicates and cancellation

Per-context16x16raw/unit covariance matrices,eigenvalues,traces,bootstrap trace intervals and norm distributions are saved in saved_signal_audit.json. Unit covariance traces are close to1while conditional mean norms are small. Between41%and49%of context vectors oppose their own mean direction: two broad groups retain extensive sign cancellation. Raw cancellation ratios within contexts range.038–.201. Thus distinct group means can be separated yet weak.

Saved independent-fit kappa and added paired-row bootstrap95%intervals:stage1.075[.056,.100],stage10.693[.678,.710],stage20.608[.588,.624],stage29.808[.791,.821]. Stage20replication is borderline at the historical.6point threshold. Different simulator replicates here mean independent joint-law trajectories; they are not repeated posterior draws for one fixed prefix.

With the same ridge feature family, held-out RAW-gradient MSE difference against a constant training mean has95%intervals spanning0at stages1,10,29and is positive at stage20[.000823,.002401]. Predictability of normalized directions therefore does not establish predictability of actual unnormalized learning gradients. Heavy tails and feature/regression misspecification remain alternatives to intrinsically weak conditional demands.

## F, G, J: noise sources, stage and reference dependence

Crossed diagnostics repeat contrasts on the same complete trajectory, and repeat observation noise at fixed generating parameters on freshly recomputed trajectories. The latter changes observed histories and is NOT a fixed-history posterior estimate. Estimated contrast contributions to raw-gradient variance are75–81%; residual joint-law variability remains. A small observation decomposition leaves positive residual noise/path components, but the stage1truth component is negative after finite-replicate correction: this is an imprecise method-of-moments estimate, not negative physical variance. Components must not be interpreted as a complete Var(G|observed history) decomposition.

The four stages have29,20,10and1remaining decisions. Stage29is most improved by contrast averaging; stages10/20retain weak coherence despite more stable assignments. No general horizon-invariant specialization is established.

No new reference training was done. At matched exogenous draws, an initial shared path and the existing600-update native-control path under routed masking were compared with shared600. In physical latent-action adjoint coordinates, native600-mask mean directional dots versus shared600range.466–.657; initial-shared versus shared600range-.017–.043. Reference path/training matters. Histories differ and latent feature coordinates can rotate, so these comparisons do not prove failure of one fixed-history context law. Intermediate preparation checkpoints were not saved; a full training-stage learning curve cannot be reconstructed.

## Limits and correctness

V3 precision checks showed at most4.6e-6relative full-graph gradient difference and current float64/native-gradient tests pass. No numerical error explaining the weak signal was found. The pairwise normalized-vector-dot formula in this NEW analysis was corrected after collection to account for vectors below the normalization floor; the collector, data, original v3 metrics, thresholds and scientific gates were unchanged. The collection-time analysis version and source hashes are preserved.

Correctness suite:32CPU tests on UbuntuPC, including exact native objective, full-history derivatives, raw-gradient averaging identity, Gaussian sign symmetry, contrast-independent histories, deterministic draws, noncommuting normalization, batch scaling and all earlier v3/native tests. LabPC performs independent CPU validation, checksum and reproduction checks. No specialist or online optimizer is implemented.

Interpretation: estimation noise is substantial; remaining weak means cannot yet be uniquely attributed to intrinsic conditional variability, poor groups, or model misspecification. No valid fixed-prefix posterior estimator was constructed.
