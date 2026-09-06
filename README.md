# Objective-Driven Sequential BOED

This project applies sequential Bayesian optimal experimental design (sBOED)
to SIR ODE and IEEE power-grid models. It compares DAD, RL-sBOED, Step-DAD,
Myopic, Fixed, and Random designs using expected information gain (EIG) or
mean objective cost of uncertainty (MOCU).

## Installation

Create the Python environment and install the dependencies:

```bash
conda create -n mocu_optimized python=3.10 -y
conda activate mocu_optimized
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CUDA is recommended for power-grid bank generation and full experiments.

## Running experiments

Use `run.sh` for one self-contained run. It checks its required data banks,
generates missing banks, trains local models, and writes the configuration,
models, logs, and evaluation results into one folder under `experiments/`.

```bash
bash run.sh --config configs/ieee9_mocu.yaml --experiment_type objective_based \
  --T 5 --N_obs 5 --noise_sigma 0.005 --seed 101
```

Use `sweep_run.sh` for a series of independent `run.sh` executions:

```bash
bash sweep_run.sh --configs ieee9_eig --experiment_type eig_based \
  --T 3,4,5,6,7 --N_obs 5 --noise_sigma 0.005
```

Publication sweeps default to training seeds `101,202,303` and evaluation
seeds `1001,1002,1003,1004,1005`. Each run owns its models and results; model
files are not shared between experiment folders.
