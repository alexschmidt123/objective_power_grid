# V5-A training status

## 1. Repository & environment

UbuntuPC TEMP `/home/qiqi/Documents/objective_power_grid_temp`, HEAD6a592c8d2ad10eb890f1d351e579ee4129939227 with pre-existing edits preserved; RTX5080,PyTorch2.11/CUDA12.8. Existing staged V4 CPU parity and hash receipts reused. Active benchmark is location-finding,T30,standard-normal two-source prior,log-signal Gaussian sigma0.5,bounds[-4,4]^2. No power-grid/SIR benchmark changes. New timestamped folder `/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10312026_073000_moe_sboed_v5_train` follows actual native naming instead of the suggested undated alias.

## 2. Implementation

New V5-only sampler/trainer/validation modules: src/research/moe_v5a_sampler.py,moe_v5a_training.py,moe_v5a_validation.py; native config configs/moe_sboed_v5a_train_20261031.yaml; one additive --moe-sboed-v5a-training dispatch hook. Old algorithms,V1–V4 policy source,likelihood and evaluator are unchanged.

P1 starts FROM SCRATCH,seed101,one shared+seven routed/top3,8434parameters. Main full-horizon sPCE uses effective B2000,L2000,Adam5e-5,betas(.8,.998),decay.98/1000,clip5,balance.001,max50000updates,14400charged seconds. B128 gradient accumulation preserves the original whole-batch objective and global detached balancing frequency; a truth-only pass establishes that frequency, with all replay work charged. No batch/contrast reduction in P1.

V5-A alpha0.10 is the single user-predeclared choice. Each update samples one primary case and one uniform stage;512-particle,8-move SMC uses only its realized past prefix and supplies128negative draws. The selected conditional score is multiplied by30 for an unbiased stage-sum SELECTION estimator. The finite posterior particles are correlated and inference gradients are detached: this is an explicitly heuristic auxiliary, not a certified EIG bound or unbiased full-policy gradient. V5-B refinement remains OFF. Original V4 utility routing every100updates is retained with its original5percent primary-router gradient budget, not relabeled V5-B.

## 3. Correctness

Fixed128-history,two-independent-pool validation PASS in 20.93s. Coordinate SBC/rank KS=[0.052734375, 0.0673828125, 0.0576171875, 0.0615234375]; predictive KS=0.1225; predeclared maximum0.20. No failed prefixes. Source-swap support,proposal/weights,pre-resampling ESS,unique particles and acceptance diagnostics are stored. These empirical checks do not certify exact/independent posterior samples.

27CPU tests PASS,including prior23and4new tests. The first regression run exceeded the test-only16-trajectory counter; the small equivalence test batch was reduced to4 and rerun, while the actual B2000/L2000 CUDA check stayed unchanged. Gaussian tests explicitly measure detached-sampling gradient bias. Actual CUDA objective check PASS: main sPCE7.525852,saturation fraction0.906,nonzero conditional gradients in shared/router/selected routed/head; peak allocation0.865GB. Original gate semantics remain unnormalized selected probabilities; no architecture change.

P0 native CUDA smoke completed3updates. A separate controlled interruption/resume restored model,Adam,scheduler,RNG and prior charged time,then completed3cumulative updates in a new output identity. P1 source/accounting was finalized before launch; phaseP0weights are not reused in P1.

## 4. Diagnostics

Historical L2000 saturation91.42percent/91.71percent remains confirmed. New conditional scores are signed,not clipped. The CUDA diagnostic produced a nonzero auxiliary gradient even with90.6percent main trajectories near the ceiling. A single-case stage estimator has high variance; its auxiliary norm exceeded the main norm in that diagnostic. Alpha was NOT tuned afterwards; native total-gradient clipping remains5. This is a research risk,not proof of efficacy.

Useful V5-A expert specialization is NOT MEASURED while training runs. Existing V4 tiny L32 route differences were zero/inconclusive. V5-B stays OFF and is not a gate for this run. No semantic expert roles or specialization success is claimed.

## 5. Resources

Live primary state at report generation: PID1736276,supervisor1736275,954completed updates,1908000primary training trajectories,1738.46charged seconds. Current actual simulator/likelihood particle-stage counter115464545030; conditional wall total154.27s. Main fresh draws,truth-only frequency replay,conditional replay,SMC proposals/likelihoods and native V4 utility all count inside the charged training clock. GPU lease remains exclusive while CPU SMC executes. No excluded trainable phase.

compute_accounting.csv is the supervisor-maintained live P1 row; readiness_compute_accounting.csv separately records CPU validation and discarded GPU correctness/smoke phases. These readiness weights were not transferred into P1. Checkpoints include model,optimizer,scheduler,RNG,source/config hashes and charged time; no best-EIG selection. A hard parent guard acts only on its own worker group before14400seconds. Hardware differs from LabPC RTX4090; equal nominal wall allowance is NOT equal FLOPs or identical training work.

## 6. Performance

V5-A EIG PENDING. Training sPCE is not held-out EIG. Frozen reference V4 seed10110.98248,seed10211.08123; native vsOED seed10112.48085±0.04168nats. After the final checkpoint freezes,the existing explicit-likelihood CommonValidation scorer will use the preserved2000cases/1Mcontrasts/seed1001 bank,with paired V5-A−V4 and V5-A−native-vsOED seed101 uncertainty. Cases/bank hashes were verified; reserved8831001was not used. Seed102native-vsOED is not verified and will not be substituted by seed101. No new V5 EIG value or multi-seed improvement is asserted.

## 7. Blockers & next steps

State RUNNING at the recorded snapshot. No current correctness blocker. Seed102NOT STARTED; ready native command is in train_commands.sh and next_commands.sh. Seed103–105commands are prepared/commented only. No unattended second training job.

Monitor: `ssh ubuntu-pc 'cat /home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10312026_073000_moe_sboed_v5_train/run_status.json'`. Primary stdout: `/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10312026_073000_moe_sboed_v5_train/train_seed101.log`; native metrics/checkpoints under `/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10312026_073000_moe_sboed_v5_train/seed101/`. Post-training paired evaluation is in the same worker,not an overlapping GPU job. If it fails,training is preserved and evaluation_pending.json gives state; use the real evaluate action with the final checkpoint path from status. Resume of a graceful interruption uses --resume-checkpoint into a NEW output identity and credits all prior stopped-clock time; an abrupt unaccounted interval fails closed.

PROMOTION NOT APPLIED. LABPC UNCHANGED (only completed immutable reference reads). NO GITHUB PUSH. The run is a developmental approximate-posterior V5-A candidate; no guaranteed EIG win or expert specialization claim.
