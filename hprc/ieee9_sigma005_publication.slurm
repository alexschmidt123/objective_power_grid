#!/bin/bash
#SBATCH --job-name=ieee9-s005-pub
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/ieee9-s005-%A-%a.out

source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/12.1.1

ROOT=/scratch/user/g.lin/Documents/objective_power_grid
RUNTIME=/scratch/user/g.lin/objective_power_grid_runtime
ENV_DIR=/scratch/user/g.lin/venvs/objective_power_grid
CAMPAIGN=ieee9_pub_sigma005_20260905
cd "$ROOT"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$RUNTIME/manifests" "$RUNTIME/slurm"

CELLS=(
  "eig 3 101" "eig 3 202" "eig 3 303"
  "eig 4 101" "eig 4 202" "eig 4 303"
  "eig 5 101" "eig 5 202" "eig 5 303"
  "mocu 3 101" "mocu 3 202" "mocu 3 303"
  "mocu 4 101" "mocu 4 202" "mocu 4 303"
  "mocu 5 101" "mocu 5 202" "mocu 5 303"
)
read -r KIND T SEED <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
if [[ "$KIND" == "eig" ]]; then
  CONFIG=configs/ieee9_eig_hprc.yaml
  TYPE=eig_based
  SUFFIX=EIG
else
  CONFIG=configs/ieee9_mocu.yaml
  TYPE=objective_based
  SUFFIX=Uctrl
fi
STAMP=$(date +%m%d%Y_%H%M%S)
EXP_DIR="$RUNTIME/experiments/${STAMP}_ieee9_hprc_seed${SEED}_${SUFFIX}_T${T}_Nobs5_sigma0p005"
test ! -e "$EXP_DIR"
printf '%s\n' "$EXP_DIR" > "$RUNTIME/manifests/${CAMPAIGN}_task${SLURM_ARRAY_TASK_ID}.path"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) kind=$KIND T=$T seed=$SEED"
nvidia-smi

bash run.sh \
  --config "$CONFIG" \
  --experiment_type "$TYPE" \
  --T "$T" \
  --N_obs 5 \
  --noise_sigma 0.005 \
  --seed "$SEED" \
  --eval-seeds 1001,1002,1003,1004,1005 \
  --exp-dir "$EXP_DIR"
