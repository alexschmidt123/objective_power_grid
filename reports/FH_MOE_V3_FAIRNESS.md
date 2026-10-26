# FH-MoE-v3 fairness and compute

Only LabPC seed101 has a14400-second optimization allowance. All trainable backbone preparation, routed specialization, router and joint optimization must fit inside it. Ubuntu seed102 must match its completed optimizer count with identical promoted source/configuration, batch, contrasts, optimizer, update schedule and precision where supported; there is no independent four-hour cutoff. Matching updates does not eliminate hardware/software/rounding effects. Neither main run was launched.

The discovery pilot was extra scientific development, not four-hour compute-matched training. No pretrained weights initialized either arm. Each of two arms completed600updates,1200000 trajectories, B2000,L2000,H30 and72036000000 scalar conditional likelihood evaluations. Both use native Adam5e-5,betas(.8,.998),clip5,float64 and LR decay.98 afterupdate1 and each1000updates thereafter. Together:1200updates,2400000training trajectories and144072000000likelihood evaluations. Offline discovery adds24000complete trajectories; the float32/float64 gradient probe adds256 trajectories. Development evaluation adds512trajectories per policy with262144contrasts.

| Measured UbuntuPC task component | Seconds |
|---|---:|
| Both preparation arms, combined | 667.753737 |
| Discovery, including precision probe | 3.115003 |
| Independent development evaluation | 5.993277 |
| Entire GPU-locked discovery task including setup/checkpoint IO | 677.504222 |
| Main LabPC optimization | 0 |
| Main UbuntuPC replication | 0 |

GPU:NVIDIA RTX5080; peak allocated GPU memory11202646016bytes. Wall time under exclusive device lock is reported, not profiled GPU-kernel-active time. CPU process seconds for the pilot were681.236850 and overlap GPU task wall time; do not add them to wall time. CPU tests, source inspection, integration, file transfer and report/diagram generation are additional preparation costs; no precise complete end-to-end CPU/GPU total was metered. The677.5-second task subtotal is not an exact entire research-project cost.

Historical protocols were audited in the v2 branch records. Original v1:29198updates,14398.167729charged seconds onRTX4090, B2000,L2000,H30. Historical FH-v1:26335/20575updates under separate four-hour budgets; this differs from the proposed v3 matched-workload replication. vsOED's reported6805.638266training seconds completed its planned schedule within14400 allowance. RL-sBOED reported14398.285173; ScoreBED13698.721751 with score and policy components. These are native methods with different work and schedule endpoints, not equal-update controls. The available dense control has8418parameters versus8434MoE and different throughput/update count; no causal v3-versus-dense conclusion exists.

No source/configuration was altered during the running pilot. Local device lock used ~/.cache/objective_power_grid/device_0.lock. Dirty historical trees and old checkpoints remain untouched. No official bank was used for model selection or method tuning. LabPC is the sole publishing machine; neither main nor any historical branch is merged automatically.
