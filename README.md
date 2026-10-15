## Introduction

**MoE-sBOED: Mixture-of-Experts for Sequential Bayesian Optimal Experimental Design** learns adaptive measurement policies offline by maximizing full-horizon expected information gain (EIG). Policies use observed design–observation histories, four independent experts, and DSelect-1 sparse routing. Training uses sequential prior contrastive estimation (sPCE); parameter-estimation MSE is a secondary evaluation metric.

The family contains four configurable variants:

| Method | Variant ID | Architecture |
|---|---|---|
| MoE-sBOED | `moe` | Four full-policy experts with per-step routing |
| ResMoE-sBOED | `resmoe` | Pretrained DAD-MLP base plus four residual experts |
| SwitchMoE-sBOED | `switchmoe` | Four full-policy experts with learned continuation and rerouting |
| SwitchResMoE-sBOED | `switchresmoe` | Pretrained base plus four residual experts with learned switching |

SwitchMoE-sBOED retains compatibility with historical Strategy-MoE checkpoints. Location finding is the primary four-variant benchmark. SIR ODE supports validated model rollouts; residual SIR performance experiments require a compatible pretrained base. Existing power-grid experiments remain available through their benchmark configurations. Comparisons include DAD and the stronger DAD-MLP single-policy control; architectural changes alone do not establish an EIG advantage.

See the [framework report](docs/moe_sboed_framework.md) for architecture details, compatibility checks, diagnostics, and experiment commands.

## Install

Clone the repository and create the reference Linux environment with Python 3.10 and CUDA 12.1:

```bash
git clone https://github.com/alexschmidt123/objective_power_grid.git
cd objective_power_grid
conda create -n mocu_optimized python=3.10 -y
conda activate mocu_optimized
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit -y
python -m pip install -r requirements.txt pytest
```

Install the GPU-specific PyTorch build before the remaining requirements. The [requirements file](requirements.txt) also documents the CUDA 12.8 environment for newer GPUs. The optional ScoreBED baseline uses a separate Python 3.11 JAX/Flax environment; see its [vendored documentation](src/research/vendor/scorebed/README.md).

## Run

Run commands from the repository root. Start with a location-finding smoke test:

```bash
bash run.sh --config configs/location_finding_eig.yaml --variant moe --smoke
```

Choose one variant per full experiment:

```bash
bash run.sh --config configs/location_finding_eig.yaml --variant moe --seed 101 --eval-seeds 1001
bash run.sh --config configs/location_finding_eig.yaml --variant switchmoe --seed 101 --eval-seeds 1001
```

Residual variants load a complete pretrained DAD-MLP checkpoint. Keep it in its original run directory with `run_config.json` and `models/sum_single_training.json`. Model checkpoints and experiment outputs are stored locally rather than committed to GitHub.

```bash
BASE=/absolute/path/to/dad_mlp_run/models/sum_single.pt
bash run.sh --config configs/location_finding_eig.yaml --variant resmoe --base-checkpoint "$BASE" --seed 101 --eval-seeds 1001
bash run.sh --config configs/location_finding_eig.yaml --variant switchresmoe --base-checkpoint "$BASE" --seed 101 --eval-seeds 1001
```

The base is frozen by default. Full location runs use one million additional primary training episodes, training contrasts L=1,024, and common held-out evaluation on 2,000 trajectories with L=1,000,000. Residual training reports the base pretraining effort separately. Evaluation seed 1001 is development data.

Train a DAD-MLP base or run other benchmark smoke tests:

```bash
bash run.sh --config configs/location_finding_eig.yaml --method sum_single --seed 101 --eval-seeds 1001
bash run.sh --config configs/sir_ode_research.yaml --variant switchmoe --T 3 --smoke
bash run.sh --config configs/ieee9_research_eig.yaml --method dad --T 3 --smoke
```

Evaluate a saved family checkpoint without retraining, using its matching variant and benchmark:

```bash
bash run.sh --config configs/location_finding_eig.yaml --variant switchmoe --family-evaluate-from /absolute/path/to/switchmoe.pt --eval-seeds 1001
```

Run the core architecture and equivalence tests:

```bash
python -m pytest -q tools/tests/test_moe_family.py tools/tests/test_strategy_moe.py
```

Results, configurations, source snapshots, checkpoints, and diagnostics are saved under `experiments/MMDDYYYY/MMDDYYYY_<campaign>/`. Use `bash run.sh --help` for all options and `sweep_run.sh` for explicit experiment sweeps.
