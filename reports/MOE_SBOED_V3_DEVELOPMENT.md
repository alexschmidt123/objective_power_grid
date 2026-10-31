# Predeclared development pilot

Status DEVELOPMENT_UNVALIDATED. No STABLE integration or historical checkpoint changes. Seed303; location horizon30; native B2000/L2000, Adam5e-5 betas(.8,.998), gradientclip5, gamma.98 after updates1,1001,... . Native v1 is freshly trained from its unchanged policy code, not substituted with an official long-run checkpoint. Five arms execute sequentially in milestone blocks: v3, v1, parameter-matched dense, shared/no-routing, v3-noaux. Milestones1024,2048,4096,8192, stopping each arm at7200 charged seconds or finite schedule, whichever first. Latest completed common milestone is selected by compute eligibility, never highest EIG.

Milestone evaluation N256, contrasts65536, seed97402871. Final common milestone N512, contrasts262144, seed97402872. Independent confirmation only for a positive paired lower95% bound, seed97402873. All use paired truth/noise/contrast cases, preserved NPZ and SHA identities. No official seed1001 randomness. Different contrast resolutions define different finite estimators, not different training objectives. No historical official EIG is used for candidate selection.

All startup/CUDA preparation and per-arm model/optimizer setup, simulation, auxiliary fitting, optimization and checkpoint serialization are charged. Shared startup is conservatively charged in full to each arm. Final evaluation and diagnostics are separately metered. No automatic failure restart. First required native cycle and safe update estimate are recorded; dynamic cutoff reserves serialization time. No SIR/grid v3 training.

Decision: measured negative/inconclusive results are preserved, no automatic v4. A replicated development gain requires subsequent independent scientific review before stable promotion or full official training. One training seed cannot establish benchmark superiority.


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
