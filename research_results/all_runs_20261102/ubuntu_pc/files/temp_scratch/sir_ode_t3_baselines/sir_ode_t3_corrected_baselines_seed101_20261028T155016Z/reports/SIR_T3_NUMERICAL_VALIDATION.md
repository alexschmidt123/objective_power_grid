# T3 numerical validation

CPU regression:16passed,3GPU checks skipped with CUDA deliberately hidden. Separate GPU-locked suite:7passed, including all-input VJPs against the original fused RK4 on257 heterogeneous cases. Initial eager/fused independent state/likelihood/action/parameter checks gave maximum errors about5.7e-14 in state and2.9e-11 in parameter sensitivity. Float64FMA/per-block padding candidate passed the same checks. Independent SciPy DOP853 checks use rtol1e-12,atol1e-13, infected/state absolute error<.001, small compared with observation sigma1.

ScoreBED JAX/Torch T3 mean and action-gradient equivalence passed on independent four-trajectory oracle cases. See profiling/scorebed/result.json for exact errors/timings. This validates the common physical sampler, not a trained score model.

Integration checks passed: independent terminal likelihood replay versus the new streamed paired scorer for Random,Fixed,DAD,vsOED; checkpoint portability; full native RL runtime serialization; full native vsOED actor/5critic/5posterior cycles; ODE/sensitivity/likelihood component timings. Fixed shared-bank objective AND gradient agree with full replay on identical shared particles. Its expectation matches finite-L sPCE for each outer trajectory because contrasts remain iid prior independent of that truth/noise; sharing changes cross-outer covariance. No interpolation/approximate trajectory cache is used.

Test-only CPU-device launch mismatch was repaired before primary execution. Failed test logs are preserved. CPU receipts, GPU XML, numerical oracles, integration_tests.json and all cost probes remain in manifests/profiling.
