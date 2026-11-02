# Sum-MoE slower sharpening: seed 101

Only gamma schedule changed: 1 through 30%, cosine to 0.001 at 95%. All original checkpoints preserved.

| Policy | Hard EIG | Relaxed EIG | Training seconds |
|---|---:|---:|---:|
| baseline | 9.380161 | 9.377376 | 1588.4 |
| slower | 9.829436 | 9.836327 | 1630.6 |
| DAD reference | 9.766654 | — | — |

Paired slower minus baseline: {'mean': 0.4492753614488299, 'standard_error': 0.05898204811786823, 'conditional_95_interval': [0.3336705471378082, 0.5648801757598516], 'n': 2000}

One seed does not establish superiority across training seeds. See report.json and per-policy CSV/JSON for stage usage, training exposure and router spikes.

baseline held-out expert frequencies (first action excluded): [0.0, 0.2859655172413793, 0.5022413793103448, 0.21179310344827587]
baseline final 5% training gradient exposure >0.001: [0.0, 1.0, 1.0, 1.0]

slower held-out expert frequencies (first action excluded): [0.002310344827586207, 0.2645689655172414, 0.571051724137931, 0.16206896551724137]
slower final 5% training gradient exposure >0.001: [0.710997442455243, 1.0, 1.0, 1.0]

Post-initial hard selections: 58,000 decisions per policy.

| Expert (zero based) | Baseline usage % | Slower usage % | Baseline final 5% gradient exposure % | Slower final 5% gradient exposure % |
|---|---:|---:|---:|---:|
| 0 | 0.000 | 0.231 | 0.000 | 71.100 |
| 1 | 28.597 | 26.457 | 100.000 | 100.000 |
| 2 | 50.224 | 57.105 | 100.000 | 100.000 |
| 3 | 21.179 | 16.207 | 100.000 | 100.000 |

Gradient exposure means aggregate expert gradient norm after clipping >0.001 per minibatch; it does not establish useful specialization. The JSON also reports the 1e-8 sensitivity threshold and whole-training/final-20% exposure.

| Policy | Router raw p99 norm | Router raw maximum | Updates with raw norm >1000 (%) | Maximum clipped norm |
|---|---:|---:|---:|---:|
| baseline | 1516354.603 | 1654272927.366 | 17.2224 | 5.000000 |
| slower | 2536.836 | 73101487.840 | 2.0992 | 5.000000 |

Hard minus relaxed EIG with paired conditional Monte Carlo uncertainty:
- baseline: {'mean': 0.0027846846782840907, 'standard_error': 0.00875852181408812, 'conditional_95_interval': [-0.014382018077328623, 0.019951387433896805], 'n': 2000}
- slower: {'mean': -0.006891193462874173, 'standard_error': 0.005421735852904187, 'conditional_95_interval': [-0.01751779573456638, 0.0037354088088180325], 'n': 2000}

Slower minus DAD: {'mean': 0.06278157185860338, 'standard_error': 0.06072719341441758, 'conditional_95_interval': [-0.05624372723365509, 0.18180687095086184], 'n': 2000}

Each full training used 1,000,000 primary episodes and L=1024. Each held-out evaluation used 2,000 paired trajectories and L=1,000,000. The separate smoke/timing checks used 8/6,400 episodes, respectively. No architecture, objective, optimizer, clipping, regularization, model-size, or evaluation change was introduced.

Implementation: slower.yaml changes only moe_router_warmup_fraction=0.3 and moe_router_finish_fraction=0.95 relative to resolved baseline settings. The existing routing-schedule implementation was reused; scientific source hashes and helper AST identity were checked before launch.

Stage-wise held-out usage is in baseline/stage_usage.csv and slower/stage_usage.csv. Training-phase by experimental-stage exposure is in each training_stage_usage.csv. Complete gradient summaries and costs are in report.json. All original checkpoints and earlier results remain intact.
