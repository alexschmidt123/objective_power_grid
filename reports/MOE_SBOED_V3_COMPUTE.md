# Development compute

Maximum7200 charged training seconds per NEW arm; up to five arms, no parallel heavy GPU training. Normal exclusive locking uses BOTH STABLE/.gpu0.lock and ~/.cache/objective_power_grid/device_0.lock, then checks existing GPU processes before allocating. If busy, the worker exits GPU_BUSY_NOT_LAUNCHED rather than interrupting or preempting another experiment. No unmanaged background polling queue.

Initialization, native optimizer updates, auxiliary re-evaluation, primary trajectories, contrast likelihood workload, checkpoint IO and CUDA prep are included. CPU verification and final evaluation are separately reported. Conservative wall cutoff reserves at least30seconds; an alarm aborts incomplete iterations before the boundary and preserves the previous budget-eligible checkpoint. Partial unfinished updates are not labeled completed. No process is automatically restarted on failure.

Atomic status: scratch campaign/status.json and reports/MOE_SBOED_V3_CURRENT.json. JSONL curves per arm include update, wall time, sPCE, auxiliary NLL, gradient norm and learning rate. Checkpoint sidecars contain SHA256, counts and charged time including serialization. Frozen complete source/config hashes, source archive, Git revision/dirty state, effective configuration and seed accompany the run. Evaluation reports contain separately metered seconds and likelihood work. GPU throughput latency is batch16 per-decision throughput, not single-item latency.


<!-- observed-progress -->
Observed 2026-10-28T15:05:17Z. State **RUNNING**, active arm **v3**, PID 811813.

| Arm | Updates | Charged seconds | Fresh episodes | Scalar likelihood evaluations | State |
|---|---:|---:|---:|---:|---|
| v3 | 250 | 127.674 | 500000 | 30015000000 | TRAINING |
| v1 | 0 | 1.329 | 0 | 0 | READY |
| dense | 0 | 1.328 | 0 | 0 | READY |
| shared | 0 | 1.328 | 0 | 0 | READY |
| noaux | 0 | 1.328 | 0 | 0 | READY |

This is a timestamped observation, not a completion claim. Current atomic status: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_sboed_v3/location_seed303_20261028T150309Z/status.json`.
Source revision: `92e59f9`, with full dirty dependency tree independently frozen by source manifest `40a31e728ccde64dc05bfe9b77bf06a678b1a6076308863cc09788bb46c3fca8` and verified519-file source archive. Root dispatch and recovered dependencies are not represented by the commit alone. No push or main merge.

No matched development EIG, paired improvement, dense/shared comparison, or useful-specialization finding is available yet. Full training is not yet justified. Grid physical interfaces remain blocked by nvcc.
