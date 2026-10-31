import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';R.mkdir(exist_ok=True)
READY=ROOT/'scratch_runs/10272026/10272026_sir_fourhour_readiness'
plans=json.loads((READY/'FROZEN_NATIVE_PLANS.json').read_text())
(R/'SIR_SEED101_PROTOCOL.md').write_text('''# Corrected SIR ODE six-baseline seed101 protocol

Only Random, Optimized Fixed, corrected DAD, corrected RL-sBOED, corrected vsOED, and corrected ScoreBED are included. No generic MoE or residual/switching variant is part of this campaign. Historical one-million-episode models are supplemental; none is reused as a four-hour primary result.

Physical model: deterministic SIR ODE, population500, initially infected2, terminal time100. Independent log(beta)~Normal(log(.5),.5²) and log(gamma)~Normal(log(.1),.5²); the configuration names beta_mean/gamma_mean denote these unchanged log-normal median parameters. Observe infected counts with independent Gaussian standard deviation1 at five continuous increasing observation times, minimum spacing.1. All methods use the same likelihood, prior, physical state carry, chronological feasible map, and adaptive float64 RK4/minimum512 steps per stage. Validated fused execution preserves the solver and its derivatives; no precision/ODE relaxation is used.

Train seed101. Each trainable method receives at most14400 charged RTX4090 seconds. Charge begins before the Python training process starts, including imports, model/optimizer setup, compilation/preparation, all training simulations, pretraining, replay, score/critic/posterior/actor work, and checkpoint serialization. Only explicitly recorded validation intervals, final EIG evaluation, and latency measurements are excluded. Random has zero training allocation. Method-native update schedules remain distinct; there is no one-million-episode quota or equal-update constraint.

Native measured-unit plans are frozen before main execution in campaign metadata/FROZEN_NATIVE_PLANS.json. They use measured native unit costs, a1.25 timing safety factor, and a60second finalization reserve. The complete native schedule or the conservative compute-derived maximum may end before the deadline; all shortened native schedules are explicitly COMPUTE_LIMITED_NATIVE_IMPLEMENTATION. A timing ceiling is not convergence evidence. Save the latest fully serialized valid budget-eligible policy, never the checkpoint with the highest test EIG. Native scheduler reference horizons remain unchanged when a measured guard shortens execution.

The persistent runner uses one exclusive GPU lock, immutable source/configuration hashes, atomic status files, sequential methods, detached child process groups, and a charged-time watchdog. Failed phases retain logs and artifacts and are not automatically restarted. Primary training enters through the STABLE run.sh launcher with an exact frozen authorization. The initial queue is DAD, Random evaluation, Fixed, RL-sBOED, vsOED, ScoreBED, so actual trainable work starts before the long common final evaluation. No simultaneous heavy GPU jobs.

Final development evaluation uses2048 paired trajectories, common seed1001, and ONE frozen shared prior-contrast bank of1000000 particles. The sequential explicit-likelihood sPCE score includes the truth likelihood in its denominator. Report EIG in nats, MC standard error, paired differences and paired standard errors against DAD/vsOED, raw scores, online CPU decision latency, measured time/work, and checkpoint/source/configuration hashes. The estimate remains a finite-contrast lower bound. Neither the resolution plan nor a single training seed establishes unbiased EIG or statistical superiority. Reserved confirmation seeds98502691 and8831001 remain unused.

Independent estimator calibration uses seed97402621,256 cases each for Random and a preserved corrected historical DAD policy, nested16,384/65,536/262,144/1,000,000 contrasts. That historical policy is used only to inspect numerical resolution, not as a new primary baseline. Resolution is selected by the predeclared common candidate grid and sensitivity, never by the winning method. Residual tail uncertainty is reported explicitly. The single bounded readiness campaign is metered separately, with an aggregate1800 GPU-locked-second cap; no additional speculative gates or automatic retries are authorized.

CPU numerical/native regression results, reused identical-source GPU equivalence receipts, native timing, checkpoint/cutoff tests, and the source/configuration freeze are retained in campaign metadata. Completed finite evaluated primary results remain distinct from interrupted/failed or supplemental historical records. No automatic merge to main. LabPC alone may publish to GitHub.
''')
(R/'SIR_BASELINE_FIDELITY.md').write_text('''# SIR baseline native fidelity

All six methods share a deterministic ODE adaptation. This is not a claim of reproducing published stochastic-SIR/SDE results. Verified native algorithm operations are separated from this common physical/chronological adapter. Vendored upstream bodies remain unchanged.

| Method | Verified native algorithm and reference schedule | Adaptation and limit |
|---|---|---|
| Random | Prescribed chronological feasible random-time sampler; no optimizer | Same SIR prior, likelihood, and spacing; zero training |
| Fixed | Static full-horizon sPCE counterpart from pinned DAD README:50,000 updates,B/L2000,Adam LR1e-4,betas(.8,.998),decay.98/1000 | Five nonadaptive feasible time logits; measured four-hour truncation; no validation/test action selection |
| DAD | Pinned ae-foster/dad4b1008174e1531d1f14601d83cef481c0f586f36:sum encoder2→256→16,shared scalar emitter including empty history,full-horizon sPCE,50,000 updates,B/L2000,Adam5e-5,betas(.8,.998),decay.98/1000,no clipping | Normalized observed time/count history and differentiable chronological SIR ODE; original gradient/Adam equivalence checked |
| RL-sBOED | Pinned RL-BOED54fdc7b6d23849ab1df8fd85d42af7c8961f930e, scripts/source_experiment_id.sh:20,001 collection iterations,B100,L100000,replay capacity10M,replay minibatch4096,warm100k transitions,64 native REDQ/SAC gradient steps per collection,two critics,actor1e-4,critic3e-4,discount.9,tau.001,entropy anneal1/14000 | Two-channel observable SIR history and native tanh-action-to-feasible-time map; replay stores native action; no simplification of native kernels; earlier1M-capacity replay was a disclosed historical resource departure and is not the new primary setting |
| vsOED | Pinned wgshen/vsOED d0884f9ba30fbddc56618841c68af2f3bfe16539; native actor/critic/GMM posterior,10,001 iterations,initial replay20k,new1000,replay batch10k,five posterior/five critic steps,actor LR3e-4,posterior/critic LR1e-3,decay.9999,discount1,target.1 | Deterministic SIR stateful adapter, log-parameter posterior and common chronological percentage map; source-native rates replace historical transferred(.0005,.0005,.001) profile explicitly |
| ScoreBED | Pinned cc8e996e3ee3237ab4c9bd3ef80e2e4b4aa9cee4, verified composed double_pend native50k score/500 policy schedule,B1024,score warmup/cosine reference50k,policy LR1e-4,correct scalar-observation potential head,GRU/EMA native policy and score gradients | SIR model and exact adaptive RK4 customVJP; full100014 normalization/initialization simulations charged. A timing-derived matched fraction of50k/500 is explicitly compute limited; no historical399:1 episode ratio or generic19:1 replacement. Scheduler reference horizons stay50k/500 |

All deployed learned policies output designs directly from observed histories, without online design optimization. Score-model initialization and other expensive trainable phases are included in the charged clock. No invalid pre-correction checkpoint is reused. Exact source hashes, effective profiles, finite/native update maxima, measured timings, and residual discrepancies accompany the run artifacts. A compute-limited implementation is not paper-level convergence evidence.
''')
resolution=json.loads((READY/'resolution/resolution.json').read_text())
text='''# SIR explicit-likelihood EIG resolution

Historical audits found substantial16,384→1,000,000 shifts for Fixed/DAD (about.58/.79nats), and existing1M→4M evidence did not certify the declared.02 absolute/.01 paired resolution tolerances. Therefore16,384 is not silently retained.

One independent calibration was performed on seed97402621,256 paired cases per Random/corrected historical DAD, with nested16,384/65,536/262,144/1,000,000 contrasts. No official-seed model tuning or method-ranking criterion was used. The common primary evaluation count is frozen at1,000,000 for all six methods; residual finite-contrast bias is explicitly unresolved rather than represented as negligible. MC standard error describes trajectory Monte Carlo variability, not estimator bias or training-seed variability. Reserved confirmation randomness is unused.

| Calibration method | Contrasts | EIG | Shift to1M | Paired shift SE |
|---|---:|---:|---:|---:|
'''
for row in resolution['records']:text+=f"| {row['method']} | {row['contrasts']} | {row['mean']:.6f} | {row['difference_to_common_1m']:.6f} | {row['paired_difference_se']:.6f} |\n"
(R/'SIR_EIG_RESOLUTION.md').write_text(text)
print('SIR protocol, fidelity, and EIG resolution reports written')
