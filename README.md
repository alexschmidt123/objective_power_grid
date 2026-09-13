# Objective-Driven Sequential BOED

Sequential Bayesian experimental design for power-grid and SIR models, comparing
DAD, RL-sBOED, Step-DAD, Myopic, Fixed and Random. IEEE9 supports EIG, minimum
safe control (MSC) and finite-loss MOCU objectives. Probes have continuous
duration, fixed injection bus/amplitude and consecutive four-second recording
windows; the physical state carries between probes. Simulation runs online.

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

Use the maintained shell entrypoints. This example is a workability smoke test:

```bash
bash run.sh --config configs/ieee9_mocu.yaml --objective msc --coverage 0.9 \
  --T 3 --N_obs 5 --noise_sigma 0.005 --seed 101 --eval-seeds 1001 \
  --smoke --output experiments/ieee9_msc_smoke
```

Use `--objective mocu` for MOCU, or `--objective eig --config
configs/ieee9_eig.yaml` for EIG. Coverage applies to the MSC/MOCU terminal
controller. Output directories must be new. For a sweep:

```bash
bash sweep_run.sh --configs ieee9_mocu --objective msc --coverage 0.9 \
  --T 3,4,5 --N_obs 5 --noise_sigma 0.005 --seed 101 --eval-seeds 1001
```

Set training, evaluation and planning budgets explicitly for performance studies;
smoke tests establish workability only. Each run saves its settings, policies,
ordered observations, evaluations and completion status under `experiments/`.
No equilibrium probe or control bank is used by the new grid protocol.

IEEE14/IEEE30 physical configurations are retained, but their online backends
are not yet implemented and runs are blocked. SIR remains available through
`run.sh --config configs/sir_ode_eig.yaml --experiment_type eig_based`.
Run checks with `bash scripts/check.sh` in the activated environment.
See [AGENTS.md](AGENTS.md) for authorization, method adaptations, detailed
workflows, reporting and HPRC instructions. Manuscript sources are in `documents/`.
