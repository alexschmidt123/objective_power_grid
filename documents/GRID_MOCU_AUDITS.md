# IEEE14 and IEEE30 finite-loss audit protocol

Research entrypoints are under `tools/`; the Slurm wrapper is `hprc/grid_mocu_audit_stage.slurm`. No trained-policy run, production bank, or final test evaluation is involved.

For each system:

1. Generate an independent 512-particle planning support and 64 off-support screening systems. Preserve the configured machinewise prior. Simulate a duration pool containing the original six durations plus [0.25,0.4,0.6,0.8,1,1.25,1.5,1.75,2,2.25,2.5,2.75] at every physical bus. Store compressed observation coordinates within the audit output only.
2. Screen 16 prespecified six-duration catalogs, including the original baseline, at T=3, using three noise seeds and inexpensive planning (inner 8, outer 4, six first candidates; all second actions). Keep the catalog maximizing the lesser of mean combined and nonmyopic gains, plus the unchanged baseline. These are exploratory selections, not significance claims.
3. Check both finalists at T=3 on the same screening systems with all first actions, inner/outer 32/16 and 64/32, Fixed calibration 512 and eight restarts. Record every tier; a negative result from an unstable planner does not establish lack of DAD opportunity. Freeze finalists before fresh validation.
4. Generate 256 fresh physical systems without resampling. Compute continuous-control oracles to 1e-4 tolerance and safety at every candidate control. Verify scalar/batched oracle agreement and reproduce known probe curves. Abort and retain evidence on infeasible/nonmonotone control responses.
5. Confirm each finalist at T=3,4,5 and noise seeds 11001/11002/11003. Use all first and second actions, inner/outer 64/32, independent second-stage evaluation fantasies, and Fixed calibration 512/eight restarts. Store raw paired observations of loss, actions, controls, ESS and physical safety.
6. Report continuous-oracle realized regret u+20*(U_opt-u)_+-U_opt, posterior diagnostics and physical safety separately. Cluster repeated noise seeds by physical theta. Use 20,000 bootstrap replicates and Bonferroni-adjusted intervals over every finalist/horizon/contrast within each system (up to 18). This does not adjust a joint two-system publication claim. Repeated seeds are not independent physical systems.

Each submitted campaign uses a frozen source snapshot and manifest. Finalization depends on successful completion of all cells. These are approximate-planner audits conditional on the prior/model/support; they neither certify optimal Fixed nor provide an upper bound on DAD or full-horizon optimal planning. Finalists are selected without fresh validation. Any later tuning consumes that validation and requires new confirmation.

IEEE30 uses the disclosed synthetic dynamic extension in `IEEE30_MODEL.md`; machine-only M/K is enforced in the production simulator. IEEE14's configuration is preserved. Duration screening is bounded and not exhaustive.
