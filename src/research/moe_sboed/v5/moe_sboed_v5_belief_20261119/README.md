# New MoE-sBOED V5:offline native-belief snapshot refresh

Official names:MoE-sBOED V5 and MoE-sBOED V5-uniform. This isolated implementation supersedes the history-conditioned V5 for further development,preserving every earlier source,checkpoint and result.

## Evidence and single change

The history-conditioned actor uses a frozen inherited30-stage GMM belief provider while the native training engine updates a separate variational posterior. On64 fresh development episodes,updated native posterior minus inherited provider averaged+1.04750 log-density nats per episode across stages5/15/25,paired95% CI[.72332,1.37168]. These are proper log-density scores,NOT EIG. Training truth was used only for offline scoring. No canonical-bank data was used to select this mechanism.

New V5 refreshes only the actor/target frozen belief buffers from their OWN current native posterior between200-update training blocks,at continuation updates200,400,...1800. Both arms receive exactly the same refresh schedule. No refresh occurs after terminal2000,so the final actor has trained200 updates with its deployed snapshot. No optimizer states or actor parameter identities change during transfer. No additional posterior loss,optimizer update,simulator episode,teacher policy or imitation is introduced. Critic diagnostics are descriptive and do not justify a new critic objective;the critic remains unchanged.

The four specialist experts,the history conditioner,644→64→4 router,soft128D weighted sum,128→32→2 head,and117→16→2 bounded.2 weak branch are identical to the previous history-conditioned architecture. All four pathways remain active;single-expert competence is not a training target. No candidate grid or larger decoder is introduced. Inference uses the current frozen snapshot and observed prefix only.

## No online optimization

The deployed actor source contains NO snapshot-refresh method. Refresh is an offline training function in run.py. Actor.forward uses fixed neural inference and existing deterministic likelihood/probe computations. No optimization,posterior fitting,MCMC,rollout search,candidate refinement or parameter update occurs during evaluation or deployment.

The executable deployment audit freezes every parameter,compares all state buffers before/after,and traps autograd.grad/backward,Tensor.backward,optimizer construction,enable_grad,and random search primitives during action generation. Audited actions remain bitwise equal. Forward method sources are inspected and hashed. Fixed loops over network layers,history chunks,stages and deterministic sigma-point/probe calculations are allowed;they are not objective-driven action optimization. The same audit runs again on each terminal model. Offline importance sampling and suffix comparisons are not imported into actor.forward.

## Offline diagnostics and limitations

A finite importance reference uses32768 particles from.75 inherited-q+.25 prior,exact current-prefix likelihood,and normalized importance weights.9/12 cases had ESS≥256. This is not an exact posterior or guaranteed multimode reference. Native critic rankings were compared with32 paired suffix rollouts for9 design/weight interventions at12 fixed fresh development histories. Predicted-best and highest rollout means matched2/12;these noisy rankings and approximate posterior references do not prove a critic bug. Returns are frozen native variational terminal-minus-prefix scores,not exact conditional EIG or a new training objective.

A legacy-Numpy seed range error in diagnostic attempt1 was corrected by modulo2**32 native seeding;PCG64 development seed stayed fixed. Partial logs and development histories were preserved and reused;no duplicate development episodes were generated. A telemetry-file collision during resumed diagnostics left attempt2 GPU utilization unavailable;do not impute zero. CPU/wall summaries and original samples are retained. Future diagnostic attempts use isolated resource directories. Conservative preparation credit,including the failed attempt and handling,is charged to BOTH run caps.

Diagnostic preparation used64 full development episodes(1920 observations) plus3456 paired suffix branches(51840 new suffix observations);not formal training episodes. New architecture audit uses native critic backward but no additional optimizer step or simulator episode.

## Matched training and primary evaluation

Seed101 only. Initialize from original V4 seed101 recipe and original frozen V2 runtime16001,not the trained diagnostic runtime or a previousV5 checkpoint. Fresh learned/uniform actor,target,critic,posterior,replay,optimizer and RNG digests must match. Native discount1/use_PCE=False variational actor–critic objective is unchanged.200warmup plus1800joint actor updates,10000critic updates,300000posterior updates,2M new episodes and60M observations PER arm. Splitting into200-update continuation calls preserves existing replay,exploration decay,schedulers and native optimizer state;native train only initializes those at update0/restart,neither occurs here. No gradient masks or auxiliary rewards. Uniform router is exactly.25 and frozen/bypassed in online,target and evaluation execution.

Four-hour hard cap PER arm includes conservatively charged preparation,training,snapshot transfer,serialization and evaluation. No selective extension or silent training restart. All block checkpoints/receipts are retained. Both arms have identical update and episode budgets;actual resource differences are disclosed. Shared V2/provider pretraining and inherited1M replay are dependencies,not free from-scratch computation.

PRIMARY evaluation is ONLY the original canonical bank and frozen scorer used forV2,V4,DAD/vsOED:2000 trajectories,1M contrasts,T30,sigma.5,original prior,bounds,truth and base noise. Reuse verified saved canonical control vectors and checkpoint hashes. No new substitute primary bank,no test-bank checkpoint/hyperparameter selection. Both terminal checkpoints are evaluated once. Report per-case MCSE and paired95% intervals separately from training-seed variability. These are finite-contrast EIG lower bounds with known saturation;MCSE does not quantify contrast bias. No automatic seeds102–105 or architecture search. Higher absolute EIG over V4 is primary;routing advantage is secondary. No performance guarantee is made.
