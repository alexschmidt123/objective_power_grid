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

Use the root launcher. Begin with correctness checks and a smoke experiment:

```bash
python -m pytest -q tools/tests/test_journal_campaign.py tools/tests/test_fixed_budget_selection.py tools/tests/test_moe_family.py tools/tests/test_baseline_readiness.py
bash run.sh --config configs/journal_location_fixed_1000000.yaml --journal --variant moe --smoke --seed 101 --eval-seeds 1001
```

Run a full-budget nonresidual method with its native learner and final-checkpoint selection:

```bash
bash run.sh --config configs/journal_location_fixed_1000000.yaml --journal --methods dad --seed 102 --eval-seeds 1001
bash run.sh --config configs/journal_location_fixed_1000000.yaml --journal --variant switchmoe --seed 102 --eval-seeds 1001
```

Residual training first requires a compatible internal500k base for the same benchmark and seed. Keep the complete base run directory and metadata; it is not a separate benchmark result:

```bash
bash run.sh --config configs/journal_location_fixed_500000.yaml --journal --methods sum_single --seed 102 --eval-seeds 1001
BASE=/absolute/path/to/that_base_run/models/sum_single.pt
bash run.sh --config configs/journal_location_fixed_500000.yaml --journal --variant resmoe --base-checkpoint "$BASE" --seed 102 --eval-seeds 1001
bash run.sh --config configs/journal_location_fixed_500000.yaml --journal --variant switchresmoe --base-checkpoint "$BASE" --seed 102 --eval-seeds 1001
```

Location scoring uses2000 paired trajectories and1million contrasts. Final checkpoints are selected at the declared budget; validation is diagnostic only. Count contrastive likelihood work, validation and runtime separately. SIR configurations are `journal_sir_t5_fixed_1000000.yaml` and its500k counterpart; SIR full development is gated on readiness and estimator-resolution checks, and requires a SIR-specific base. T10 is deferred.

On the existing LabPC repository, `python -m tools.journal_campaign prepare` audits historical seed101 artifacts and prepares the dated campaign. After its fixed-budget ScoreBED smoke check, `seal` freezes source/configuration/analysis; `location` runs missing seeds101–105 sequentially and then SIR readiness. `sir` starts only the gated T5 development study. `report` regenerates CSVs and figures. Preparation never overwrites an existing campaign. The campaign tool requires the historical artifacts for reuse; standalone root-launcher commands above train fresh policies.

All configurations, commands, hashes, checkpoints, costs, failures and statistical outputs live under `experiments/MMDDYYYY/MMDDYYYY_<campaign>/`. Completed phases can be reused by the campaign controller; exact interrupted native-trainer resume is not universally supported. Never relabel a partial run as a complete budget. Model weights remain local rather than committed to GitHub.
