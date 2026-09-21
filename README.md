# Objective-Driven Sequential BOED

Sequential Bayesian experimental design for power-grid and SIR models, comparing
DAD, RL-sBOED, Step-DAD, Myopic, Fixed and Random. IEEE9 supports EIG, minimum
safe control (MSC) and finite-loss MOCU objectives. Probes have continuous
duration, fixed injection bus/amplitude and consecutive three-second recording
windows; the physical state carries between probes. Simulation runs online. For T=3, durations strictly increase with at least
a 0.01-second gap.

## Current conference study

The short conference paper targets **EIG on IEEE9, IEEE14 and IEEE30**, comparing
**DAD, RL-sBOED, Step-DAD, Myopic, Fixed and Random**. SIR-ODE is a supplementary
benchmark. MoE remains available only when explicitly selected and is outside
this paper; MSC/MOCU are preserved for later work.

Select a single method with `run.sh --method <name>` or the six grid methods with
`--methods dad,rl_sboed,step_dad,myopic,fixed,random`. Step-DAD requires its DAD
policy. A standalone DAD run uses random initialization; a joint Fixed+DAD run
can select Fixed initialization on validation. Match and report initialization
when comparing runs. Baseline execution and timing do not require MoE modules.

IEEE9 is implemented. IEEE14 EIG passed LabPC CPU/GPU and seven-method smoke validation; see
[IEEE14 readiness](docs/ieee14_eig.md). IEEE30 still requires its specified full
dynamic backend; listing it as a target does not make it runnable.

## Installation

```bash
conda create -n mocu_optimized python=3.10 -y
conda activate mocu_optimized
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

A CUDA GPU and compatible CUDA toolkit are required for the online grid runner.

## Running experiments

New continuous-grid EIG runs default to **1,024 contrasts** for training, validation and evaluation (sPCE ceiling `log(1025) = 6.93245` nats). Override with `--contrasts`; planner particles are a separate setting. Existing runs retain their saved budgets.

Use the maintained shell entrypoints. This example is a workability smoke test:

```bash
bash run.sh --config configs/ieee9_mocu.yaml --objective msc --coverage 0.9 \
  --T 3 --N_obs 0 --noise_sigma 0.005 --seed 101 --eval-seeds 1001 \
  --smoke --output experiments/ieee9_msc_smoke
```

Use `--objective mocu` for MOCU, or `--objective eig --config
configs/ieee9_eig.yaml` for EIG. Coverage applies to the MSC/MOCU terminal
controller. Output directories must be new. For a sweep:

```bash
bash sweep_run.sh --configs ieee9_mocu --objective msc --coverage 0.9 \
  --T 3,4,5 --N_obs 0 --noise_sigma 0.005 --seed 101 --eval-seeds 1001
```

Set training, evaluation and planning budgets explicitly for performance studies;
smoke tests establish workability only. Each run saves its settings, policies,
ordered observations, evaluations and completion status under `experiments/`.
No equilibrium probe or control bank is used by the new grid protocol.

IEEE14 EIG support and its validation procedure are documented in
[IEEE14 EIG](docs/ieee14_eig.md). Check the validation receipt before full runs.
IEEE30 and IEEE14 MSC/MOCU remain blocked in the continuous CLI. SIR remains available through
`run.sh --config configs/sir_ode_eig.yaml --experiment_type eig_based`.
Run checks with `bash scripts/check.sh` in the activated environment.
See [AGENTS.md](AGENTS.md) for authorization, method adaptations, detailed
workflows, reporting and HPRC instructions. Manuscript sources are in `documents/`.

Power-grid runs default to a scalar maximum absolute RoCoF observation (`N_obs=0`) with continuous probe duration and non-reset 3-second windows. Noise `0.005` is in Hz/s and is applied to the extracted peak. Positive `N_obs` explicitly selects frequency samples with noise in Hz. IEEE14 EIG uses an explicit objective override; IEEE30 remains unavailable.

Results are grouped by start date: `experiments/MMDDYYYY/MMDDYYYY_<time>_<objective_parameters>/`. Existing runs are listed in `experiments/INDEX.csv`.

### Optional additional evaluation seeds

Fresh training remains the default. To evaluate completed matching checkpoints
without retraining, explicitly select their run directory:

    bash run.sh --evaluate-from /absolute/path/to/completed/run --eval-seeds 1002,1003,1004,1005

Source settings are inherited and checked. Results go to a new dated directory
with checkpoint hashes and training-run provenance. Use the original completed
training run (models and source_snapshot must be present), not a lightweight
result-only copy. Different training settings require a fresh run.
This option does not resume training and is not enabled for sweeps.

## Independent MoE-sBOED for SIR ODE

Select `--method moe_sboed` explicitly for the SIR EIG benchmark. The independent
policy uses four experts and learned top-2 sparse routing at every stage. Training
starts from random weights, uses PPO on terminal information gain, and selects
the checkpoint by validation EIG only. It uses no baseline checkpoint, planner
labels, fixed first action, behavioral cloning, or forced design diversity.

```bash
bash run.sh --config configs/sir_ode_eig.yaml --T 3 --method moe_sboed --seed 101 --eval-seeds 1001,1002,1003,1004,1005
```

Implementation: `src/policies/independent_moe.py` and
`src/objectives/eig/independent_moe.py`. The SIR metric is finite-particle posterior
entropy reduction; it is distinct from online IEEE9 sPCE. This discrete policy
is not yet connected to continuous-duration IEEE9. Legacy MoE and MSC/MOCU code
remain available; selecting this EIG method does not activate their training.


## Direct MoE policy for continuous IEEE9 EIG

Use `--method fixed,dad,moe_sboed --moe-training-mode policy_pathwise` to
compare DAD with four independent 64-by-64 experts and a history-conditioned
soft router. Both use the same history features, feasible-duration map,
full-horizon sPCE objective, pathwise gradients, Adam and validation selection.
Including Fixed offers both policies the same newly trained sequence as an
initialization candidate; all hidden expert weights are initialized afresh.
No DAD checkpoint or imitation loss is used. Omitting Fixed gives random
initialization. This experimental mode does not establish superiority.
Training histories record routing variation and expert-proposal differences.
