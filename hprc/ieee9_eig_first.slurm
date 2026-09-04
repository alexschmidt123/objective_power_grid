#!/bin/bash
#SBATCH --job-name=ieee9-eig-first
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/first-%j.out

set -euo pipefail

module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/12.1.1

ROOT=/scratch/user/g.lin/Documents/objective_power_grid
ENV_DIR=/scratch/user/g.lin/venvs/objective_power_grid
cd "$ROOT"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "First cell generates the fresh physical bank before training."
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

bash sweep_run.sh \
  --configs configs/ieee9_eig_hprc.yaml \
  --experiment_type eig_based \
  --T 3 \
  --N_obs 5 \
  --noise_sigma 0.01 \
  --seed 101 \
  --eval-seeds 1001,1002,1003,1004,1005
