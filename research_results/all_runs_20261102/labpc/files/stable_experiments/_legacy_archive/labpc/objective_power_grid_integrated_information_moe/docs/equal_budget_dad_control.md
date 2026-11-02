# Equal-budget DAD-MLP control

The elapsed-time-matched continued DAD-MLP exceeds ResMoE on the paired development cases.

Development results only: training seed **101**, evaluation seed **1001**, 2,000 paired trajectories, L=1,000,000. Untouched seed **8831001** was not used.

| Model | EIG | 95% trajectory MC interval | Additional episodes | Additional training (s) | Total optimization (s) |
|---|---:|---|---:|---:|---:|
| Original DAD-MLP | 10.66259855 | [10.57287569, 10.75232142] | 0 | 0.000 | 729.796 |
| DAD-MLP +1m episodes | 10.56285151 | [10.46964807, 10.65605495] | 1,000,000 | 812.571 | 1542.367 |
| DAD-MLP +matched elapsed time | 10.94265537 | [10.85280054, 11.03251021] | 2,385,408 | 1736.874 | 2466.670 |
| ResMoE-sBOED | 10.84764534 | [10.76171194, 10.93357874] | 1,000,000 | 1736.832 | 2466.628 |

ResMoE minus elapsed-matched continued DAD-MLP: **-0.09501003 nats**, paired SE 0.04697298, conditional 95% interval **[-0.18707707, -0.00294300]**.

## Exact checkpoint provenance

All paths below are relative to `/home/grads/g/g.lin/Documents/objective_power_grid`.

- **Original DAD-MLP**: `experiments/10142026/10142026_moe_controlled_diagnosis/controls/models/sum_single.pt`; SHA256 `ec309d224a5c69a9b8ab9574b094f5d4225db506b2c352bbe88e4453865ac300`. Configuration: `experiments/10142026/10142026_moe_controlled_diagnosis/controls/run_config.json`. Training metadata: `experiments/10142026/10142026_moe_controlled_diagnosis/controls/models/sum_single_training.json`.
- **DAD-MLP +1m episodes**: `experiments/10152026/10152026_013344_moe_fairness_controls_seed101/continued_dad_mlp/models/sum_single.pt`; SHA256 `baba752b81233d217a6afd50c47a3f3a9118b9bffb4bdafc9dcecdd3ae023379`. Configuration: `experiments/10152026/10152026_013344_moe_fairness_controls_seed101/continued_dad_mlp/run_config.json`. Training metadata: `experiments/10152026/10152026_013344_moe_fairness_controls_seed101/continued_dad_mlp/models/sum_single_training.json`.
- **DAD-MLP +matched elapsed time**: `experiments/10152026/10152026_equal_compute_dad_control_v2/continued_dad_mlp_walltime/models/sum_single.pt`; SHA256 `9cd260dda94a7f877428535fed7976fe0a47a128a85cabd2755f3f20a6f06fc3`. Configuration: `experiments/10152026/10152026_equal_compute_dad_control_v2/continued_dad_mlp_walltime/run_config.json`. Training metadata: `experiments/10152026/10152026_equal_compute_dad_control_v2/continued_dad_mlp_walltime/models/sum_single_training.json`.
- **ResMoE-sBOED**: `experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/resmoe/models/resmoe.pt`; SHA256 `e9ac1228b83ecd459801c7802b1aa24e595b99a494307f068fae01bfa81ffcbb`. Configuration: `experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/resmoe/run_config.json`. Training metadata: `experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/resmoe/models/resmoe_training.json`.

The exact complete DAD-MLP base uses its original permutation-invariant 3→256 ReLU→16 sum encoder, fixed zero padding to a native 772-dimensional head input, 64/64 tanh policy head, separately learned first design and original bounded action mapping. The residual/router input remains193; it is not silently substituted for the base input.

## Configuration and computational budget

Common settings: batch64, full-horizon sPCE L1024, T30, Adam lr5e-5, betas(0.8,0.998), exponential decay0.98 every1000 updates, component clipping5. Standard-normal four-coordinate prior, two sources, log-signal Gaussian SD0.5, design box[-4,4]^2. Evaluator batch16, contrast chunk16384. No estimator, simulator or action transformation was changed.

| Model | Additional contrast trajectories | Additional truth-inclusive likelihood trajectories | Additional particle-stage likelihood evaluations | Total primary episodes including base |
|---|---:|---:|---:|---:|
| DAD-MLP +1m episodes | 1,024,000,000 | 1,025,000,000 | 30,750,000,000 | 2,000,000 |
| DAD-MLP +matched elapsed time | 2,442,657,792 | 2,445,043,200 | 73,351,296,000 | 3,385,408 |
| ResMoE-sBOED | 1,024,000,000 | 1,025,000,000 | 30,750,000,000 | 2,000,000 |

Base pretraining: 1,000,000 episodes, 15,625 updates, 729.796016s optimization, plus 2.354443s diagnostic validation (512 trajectories, L65536, 33,554,944 truth-inclusive likelihood trajectories / 1,006,648,320 particle-stage evaluations, no updates, not used for checkpoint selection). Add this validation time separately to the optimization totals above. Include the base cost once per end-to-end model; it was shared in the actual campaign.

The elapsed control stops at the first completed update reaching1736.8317570788786s. Its5,000,000-episode cap is only a safety limit, not its actual episode budget. The predeclared stopping rule never examines held-out EIG. Reported particle-stage counts count complete truth-inclusive trajectory likelihoods at every stage; they do not equate backward-pass FLOPs or different neural-policy costs.

| Setting | Continued DAD-MLP | ResMoE-sBOED |
|---|---|---|
| Pretrained base | Exact original checkpoint | Same exact original checkpoint |
| Trainable parameters | 58,900 original parameters | 67,406 new residual/router parameters |
| Total parameters | 58,900 | 126,306 |
| Encoder/base/first design | Fine-tuned | Frozen |
| New experts/router | None | Four independent64/64 tanh residuals, DSelect-1 |
| Residual scale/initialization | Not applicable | alpha1; small nonidentical initialization std0.001 |
| Router schedule/regularization | Not applicable | gamma1 through30%, cosine to0.001 at95%; existing1e-4 penalties |
| Optimizer state | Fresh Adam from base weights | Fresh Adam for new components |
| Learning-rate schedule | Same initial LR and decay rule; more steps when time allows | Same rule over1m additional episodes |

## Commands

Commands were executed from the repository root. Each root command includes training (unless explicitly frozen) and the common held-out evaluation. Use a fresh output path when reproducing; existing outputs remain preserved.

resmoe_evaluation_only:

```bash
bash run.sh --config configs/location_finding_eig.yaml --seed 101 --eval-seeds 1001 --eig-only --source-check-manifest /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_equal_compute_dad_control_v2/source_guard.json --variant resmoe --family-evaluate-from /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/resmoe/models/resmoe.pt --output /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_equal_compute_dad_control_v2/resmoe_eig_only_replay
```

continued_dad_training_and_evaluation:

```bash
bash run.sh --config configs/location_finding_eig.yaml --seed 101 --eval-seeds 1001 --eig-only --source-check-manifest /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_equal_compute_dad_control_v2/source_guard.json --method sum_single --dad-mlp-continue-from /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10142026/10142026_moe_controlled_diagnosis/controls/models/sum_single.pt --continuation-seconds 1736.8317570788786 --continuation-max-episodes 5000000 --output /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_equal_compute_dad_control_v2/continued_dad_mlp_walltime
```

historical_resmoe_training_and_evaluation:

```bash
bash run.sh --config configs/location_finding_eig.yaml --variant resmoe --base-checkpoint /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10142026/10142026_moe_controlled_diagnosis/controls/models/sum_single.pt --seed 101 --eval-seeds 1001 --source-check-manifest /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/source_guard.json --output /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/resmoe
```

historical_equal_episode_dad_training_and_evaluation:

```bash
bash run.sh --config configs/location_finding_eig.yaml --method sum_single --dad-mlp-continue-from /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10142026/10142026_moe_controlled_diagnosis/controls/models/sum_single.pt --seed 101 --eval-seeds 1001 --source-check-manifest /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/source_guard.json --output /home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_013344_moe_fairness_controls_seed101/continued_dad_mlp
```

Historical source-check manifests intentionally reject changed source; reproduce those commands from their frozen source snapshot or create a fresh manifest for the reviewed current code. The original equal-episode and ResMoE commands are also preserved in their campaign `commands.json` files. A frozen ResMoE evaluation through the new EIG-only option reproduces the old common evaluator within score tolerance1e-9 and action/observation tolerance1e-10 for all2,000 cases. See `eig_only_replay_verification.json` in the campaign.

## Interpretation limits

- All results use training seed101 and repeatedly used development evaluation seed1001; seed8831001 remains unused. No final/publication result is claimed.
- Intervals are mean +/-1.96 paired trajectory standard errors, conditional on fixed checkpoints and one shared contrast bank. They exclude training-seed and contrast-bank variability.
- Elapsed training includes simulation, backward/optimizer work and diagnostics. The historical residual run overlapped CUDA posterior inference; the timed control waits for an idle GPU. Equal elapsed time is not exact GPU-FLOP matching.
- The additional one-million-episode control matches simulator/likelihood counts; the elapsed-time control may use more episodes. Both views are reported without presenting them as simultaneously matched.
- Fresh Adam is used for both additional-training phases because the saved base contains weights, not optimizer moments; LR decay restarts at the same initial value.
- ResMoE freezes the pretrained encoder, base head and first design; DAD continuation fine-tunes all original parameters. ResMoE also has more parameters and explicitly adds stage to its193-dimensional residual/router input; the original native772 DAD head keeps its original sum representation without adding a new stage input. A positive result alone cannot isolate sparse routing from residual parametrization, freezing or capacity.
- Both phases use seed101 and per-update seed101+(step+1)*104729; the first million additional draws reuse the original training random stream. Counts refer to generated episodes, not guaranteed unique draws across phases.
- Frozen dominant-expert routing matches learned ResMoE within paired uncertainty (difference +0.00826178, 95% interval [-0.04730444,+0.06382801]); useful learned sparse routing is not established. Forced routing can induce out-of-distribution histories.
- Training sPCE L1024 has ceiling log1025=6.93244789; high-precision held-out EIG uses L1000000. These are different bound resolutions.
- Only final predeclared checkpoints are evaluated; no EIG-based checkpoint selection or MSE training is used. Secondary MSE inference is omitted only for the new timed run and replay; EIG replay equivalence was verified on all2000 cases.

## Implementation and correctness

The optional elapsed-budget stop is confined to `train_control` in `src/research/moe_architecture_controls.py`; ordinary episode-budget training remains available. `src/research/runner.py` exposes the duration/safety cap and optional omission of secondary inference. `tools/early_family_eig.py` loads complete native controls and validates actual episode/time budgets. `tools/report_equal_budget_dad.py` reads fixed artifacts, validates pairing and counts, and produces this report. Residual architecture, simulator, likelihood, action mapping, sPCE implementation and existing checkpoints were not modified.

**63 focused tests passed**, including forced-routing/base equivalences, bounded rollouts, gradient reachability, exact ordinary-versus-timed training for matched updates, checkpoint round trips, EIG-only equality and paired-report accounting. The full ResMoE EIG-only replay produced zero difference in all2000 actions, observations and scores; all11 pretrained-base tensors in ResMoE are exactly identical to the original base. Test log: `experiments/10152026/10152026_equal_compute_dad_control_v2/focused_correctness.log`.


## One recommended next experiment

Replicate this paired ResMoE versus continued DAD-MLP experiment for independent training seeds102 and103. Within each seed, share the same independently pretrained DAD-MLP base, freeze hyperparameters and budget rules beforehand, measure training costs without overlapping GPU jobs, and evaluate the frozen checkpoints on untouched paired cases. Report variation across complete training pipelines as well as trajectory MC uncertainty. Do not tune from the new evaluation outcomes. This is one replication experiment, not an architecture search.

The current comparison tests whether extra training explains the observed development gain. It cannot by itself demonstrate that sparse expert specialization causes that gain. All original checkpoints and results are preserved.
