# Frozen method settings

All methods: location-finding T30, common physical likelihood/prior/noise and [-4,4]^2 actions. Seeds101–105; no MoE training. Maximum14400 charged seconds per trainable method/seed. Setup, compilation, pretraining, replay, optimization and checkpoint export count; measured held-out scoring is separate. There is no extra restart allowance.

| Method | Native learning procedure / frozen settings | Schedule / compute classification |
|---|---|---|
| Random | Truncated standard-normal designs, no history dependence | No training |
| Fixed | One nonadaptive30-stage schedule; terminal joint sPCE; B64,L1024,Adam5e-5,betas(.8,.998),decay.98/1000 | Native offline reference ceiling50000updates; same recipe as seed101; terminal checkpoint, no test selection |
| DAD | Original sum256/16 history encoder; full-horizon sPCE; B2000,L2000,Adam5e-5,betas(.8,.998),decay.98/1000,no added clipping |50000updates or4h; corrected primary-episode accounting |
| RL-sBOED | Native stochastic SAC/REDQ; B100fresh,L100000,replay1M,batch4096,64gradientcycles,2critics,discount.9,tau.001,actor1e-4,critic3e-4 |20001collection cycles or4h; warm-up charged; no simplified REINFORCE substitute |
| vsOED | Native incremental GMM actor/critic/posterior;20k initialization,1k fresh,10k replay,5critic/5posterior steps,30increments |10001nativecycles or4h; posterior and replay work charged |
| ScoreBED | Native conservative-transformer score matching then native DADNet policy score-gradient learning; B1024,AR1 reference sampler,EMA policy,one policy arm per seed |Reference190000score+10000policy updates; hardware-timing-only4h plan177365+9335,explicit COMPUTE_LIMITED adaptation; native LR horizons retained. No extra restarts or omitted stage |

Method-specific batches/contrasts and objectives are disclosed, not forced equal. Equal maximum wall time is not equal FLOPs or simulator work. Neither budget exhaustion nor native schedule completion establishes convergence. Finite training-sPCE ceilings are logged. No superiority claim before common independent evaluation.

Final:2000 common paired trajectories,L1000000,seed73831031, generated only after candidate freeze. Development validation is independent(seed73831032); it never selects final checkpoints. Report within-checkpoint MCSE and between-training-seed SD/SE separately. Seed101 archival scores are not substituted for the fresh common evaluation.
