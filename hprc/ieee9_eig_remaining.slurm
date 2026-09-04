#!/bin/bash
#SBATCH --job-name=ieee9-eig-rest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/rest-%A-%a.out

source /etc/profile
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

CELLS=(
  "3 202"
  "3 303"
  "4 101"
  "4 202"
  "4 303"
  "5 101"
  "5 202"
  "5 303"
)
read -r T SEED <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"

# Avoid second-resolution result-folder collisions for equal horizons.
sleep "$((SLURM_ARRAY_TASK_ID * 3))"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) T=$T seed=$SEED"
nvidia-smi

bash sweep_run.sh \
  --configs configs/ieee9_eig_hprc.yaml \
  --experiment_type eig_based \
  --T "$T" \
  --N_obs 5 \
  --noise_sigma 0.01 \
  --seed "$SEED" \
  --eval-seeds 1001,1002,1003,1004,1005
