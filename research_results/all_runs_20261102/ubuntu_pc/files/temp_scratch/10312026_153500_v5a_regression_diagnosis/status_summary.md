# (1) Repository & environment
UbuntuPC TEMP only; HEAD6a592c8d2ad10eb890f1d351e579ee4129939227, existing dirty state preserved. Python/PyTorch/CUDA versions inprovenance.json; RTX5080. Staged immutable V4 used; no LabPC access. New outputs only/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10312026_153500_v5a_regression_diagnosis.

# (2) Implementation
No production changes. Saved-run audit, actual prefix/surrogate tests,6-prefix36-population posterior diagnostic,16 batches/model gradient probes,4 native-size batches/model,4 on/off Adam-clone comparisons/model. No deployed architecture change. No new formal run. One32-update/arm bounded local diagnostic completed.

# (3) Correctness
30 CPU tests passed; earlier2 failures were restrictive accounting-plugin caps, preserved. Production main-loss gradient equivalence passed both checkpoints. Actual prefix/current-future-noise check passed. Stopped-sampler directional derivative abs.error1.49e-8; does not verify intended full-EIG gradient. Checkpoint/data hashes and saved EIG arithmetic verified; immutable hashes pass=True.

# (4) Diagnostics
NativeB2000 V5 weightedaux/main ratios2.87–80.41,median13.42;4/4auxdominates,3/4conflict. B16 V5conflict5/16,mean routing-set change3.28% in Adam probes. Posterior reference approximate; insufficient to certify mode coverage. Smaller update exposure is a major confound. No useful specialization established.

# (5) Resources
New diagnostic GPU lease≈30.80s, plus131.73s paired diagnostic lease (32updates/arm, not formal training).16discarded Adam clone steps total across both models; nativeB2000 gradients make zero optimizer steps. CPU posterior21.80s plus prefix/tests/reporting. Compute counters separate from historical3h59m; all costs inCSV.

# (6) Performance
NO NEW FORMAL EIG. Reused V5-A9.76586593±.04333000; pairedV4difference−1.21661020±.05330844,CI[−1.32109474,−1.11212565]. Historical V4 updates29464 versusV5 updates7714; not equal-exposure causal comparison. No statistical superiority or specialization claim.

# (7) Blockers & next steps
Strong proximal interference evidence; causal EIG attribution remains INCONCLUSIVE. One32-update/arm same-Ubuntu, same-initial-state OFF/ON pilot completed in131.73s; no formal experiment. Historical V4 host UNVERIFIED; nativevsOED specific staged reference explicitly recordsLabPC. Equal hours are not equal work. See corrected accounting. Missing historical dev/routing traces and exact posterior oracle limit conclusions. SEED102 NOT STARTED. PROMOTION NOT APPLIED. LABPC UNCHANGED. NO GITHUB PUSH.

Hardware-normalized primary work, optimizer counts, individual-host throughput and UNAVAILABLE fields are in compute_accounting.csv. Pilot is episode/update matched, NOT compute matched. NO NEW FORMAL EIG. SEED102 NOT STARTED. PROMOTION NOT APPLIED. LABPC UNCHANGED. NO GITHUB PUSH.
