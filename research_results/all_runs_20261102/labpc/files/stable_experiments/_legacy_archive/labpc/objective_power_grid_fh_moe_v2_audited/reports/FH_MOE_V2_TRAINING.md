# FH-MoE-v2 training and gate

No seed101 primary or seed102 replication run was launched. The scientific pilot failed, so the instructed eight-or-more-hour main workflow was not entered.
The revised protocol is retained: ONLY LabPC has the14400-second limit; Ubuntu matches exactly its final completed optimizer count without a four-hour limit.
All trainable backbone/specialist/router/joint updates would be charged inside the LabPC primary allowance. An18000-update secondary checkpoint is conditional on reaching it within that allowance.

Pilot: fresh seed98102601,300updates per arm,B2000,L2000,H30,original Adam5e-5/(.8,.998),update-based .98 schedule,global clip5,float64.
No pretrained policy initializes either arm. Both use identical initialization and paired simulation draws.
Total pilot wall time including independent diagnostics/evaluation:408.874871s.
Successful paired pilot native forward work:72036000000 parameter-stage propagations.
Repeated stress test:39.263384s,30updates per arm,peak11333076992bytes.
Pilot-state checkpoint remains/home/qiqi/Documents/objective_power_grid_fh_moe_v2/experiments/10262026/10262026_fh_moe_v2_development/models/pilot_state.pt and is explicitly NOT an official candidate.

Two implementation failures were preserved: CPU/CUDA default-device mismatch and repeated-update graph-retention OOM.
Corrections retain exact gradient mathematics and do not change scientific thresholds or use performance-based retries.
The OOM attempt has only bounded exact-work accounting because its last saved paired progress was update10; do not claim a precise unlogged update count.
Recorded successful-task subtotal:478.742841s across the two machines; failed OOM wall approximately14.656s, plus separately untimed earlier smokes and CPU development.
This subtotal is NOT a complete exact end-to-end cost or a four-hour-matched result.
Full ledger:fh_moe_v2_evidence/compute_ledger.json.
Actual GPU-kernel busy time was not profiled; reported GPU tasks use local exclusive-device wall timing.

No official or confirmation bank was consumed for FH-MoE-v2. No seed1001 performance was used for candidate tuning.
No unused confirmation bank was generated after a failed gate.
