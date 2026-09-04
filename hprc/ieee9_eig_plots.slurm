#!/bin/bash
#SBATCH --job-name=ieee9-eig-plots
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/plots-%j.out

source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0
module load Python/3.10.8

ROOT=/scratch/user/g.lin/Documents/objective_power_grid
ENV_DIR=/scratch/user/g.lin/venvs/objective_power_grid
cd "$ROOT"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT"

bash sweep_run.sh \
  --configs configs/ieee9_eig_hprc.yaml \
  --experiment_type eig_based \
  --T 3,4,5 \
  --N_obs 5 \
  --noise_sigma 0.01 \
  --seed 101,202,303 \
  --eval-seeds 1001,1002,1003,1004,1005 \
  --plots-only
