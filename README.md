## Introduction

MoE-sBOED studies sparse expert management for sequential Bayesian optimal experimental design. The four variants are **MoE-sBOED** (`moe`), **ResMoE-sBOED** (`resmoe`), **SwitchMoE-sBOED** (`switchmoe`) and **SwitchResMoE-sBOED** (`switchresmoe`). They retain four independent64/64tanh experts, a sum history encoder, DSelect-1 routing and full-horizon sPCE. Historical Strategy-MoE names remain checkpoint aliases.

The journal study compares exactly ten methods: Random, Optimized Fixed, DAD, simplified RL-sBOED, vsOED, ScoreBED and the four variants, on location finding and SIR ODE. Each trained method receives **one million total primary training episodes**, including pretraining. Residual models use a seed-specific500k base plus500k residual episodes; the exact frozen base is shared within each seed. The internal base is not a DAD-MLP benchmark method. Native baseline losses and architectures are retained, with resource-saving adaptations disclosed. Results do not guarantee a MoE advantage.

See the [fixed-budget journal protocol](docs/journal_fixed_budget_protocol.md), [baseline audit](docs/baseline_audit.md), and [architecture report](docs/moe_sboed_framework.md). Historical2M residual and5M vsOED runs remain separate from the primary comparison. Evaluation1001 is development data;8831001 is reserved.

## Install

Use the reference Python3.10/CUDA12.1 Linux environment:

```bash
git clone https://github.com/alexschmidt123/objective_power_grid.git
cd objective_power_grid
conda create -n mocu_optimized python=3.10 -y
conda activate mocu_optimized
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit -y
python -m pip install -r requirements.txt pytest
```

ScoreBED uses the separate `.venv_scorebed` JAX/Flax environment. Its pinned source and installation documentation are in [vendor/scorebed](src/research/vendor/scorebed/README.md). See requirements.txt for other supported CUDA builds.

## Run

Start a single run or a sweep from the repository root with an explicit config.
On Sapelo2, execute the launcher inside a Slurm allocation with the project Conda environment active.

```bash
bash run.sh --config configs/power_grid/ieee9_research_eig.yaml --method dad
bash sweep_run.sh --configs configs/power_grid/ieee9_research_eig.yaml,configs/location_finding/location_finding_eig.yaml --method dad --T 3,5
```

New run directories are direct children of `experiments/`, for example
`11232026-0930-ieee9_research_eig_T3-dad`. The project layout, collision rule,
and launch conventions are in [AGENTS.md](AGENTS.md).
Historical source, configurations and reports retain their original path aliases.
Official result registration continues to use its frozen verification procedure.

The completed five-seed development results, instability audit and original-setting selected-repeat commands are in [research records](docs/research_records/20261016/README.md). The ScoreBED validation bounds fix does not establish stable training; the failed gamma0.01 trial is archived and the default remains0.001.
