#!/bin/bash
#SBATCH --job-name=sir-eig-migrate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/sir-migrate-%A-%a.out

source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/12.1.1

ROOT=/scratch/user/g.lin/Documents/objective_power_grid
RUNTIME=/scratch/user/g.lin/objective_power_grid_runtime
ENV_DIR=/scratch/user/g.lin/venvs/objective_power_grid
CAMPAIGN=sir_ode_eig_migration_20260904

cd "$ROOT"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$RUNTIME/manifests" "$RUNTIME/slurm"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
nvidia-smi

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    # Models for T=4, seed=303 were trained on labpc. Re-run all five fixed
    # evaluation seeds on HPRC so the migrated cell has a complete evaluation.
    EXP_DIR="$RUNTIME/experiments/09042026_135537_sir_ode_EIG_T4_Nobs1_sigma1"
    printf '%s\n' "$EXP_DIR" > "$RUNTIME/manifests/${CAMPAIGN}_task0.path"
    for EVAL_SEED in 1001 1002 1003 1004 1005; do
        python3 -m src.experiment evaluate \
          --config configs/sir_ode.yaml \
          --experiment-type eig_based \
          --N_obs 1 \
          --noise_sigma 1 \
          --seed 303 \
          --eval-seed "$EVAL_SEED" \
          -T 4 \
          --exp-dir "$EXP_DIR"
    done
else
    SEEDS=(101 202 303)
    SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID - 1))]}"
    STAMP=$(date +%m%d%Y_%H%M%S)
    EXP_DIR="$RUNTIME/experiments/${STAMP}_sir_ode_hprc_seed${SEED}_EIG_T5_Nobs1_sigma1"
    test ! -e "$EXP_DIR"
    printf '%s\n' "$EXP_DIR" > "$RUNTIME/manifests/${CAMPAIGN}_task${SLURM_ARRAY_TASK_ID}.path"
    bash run.sh \
      --config configs/sir_ode.yaml \
      --experiment_type eig_based \
      --T 5 \
      --N_obs 1 \
      --noise_sigma 1 \
      --seed "$SEED" \
      --eval-seeds 1001,1002,1003,1004,1005 \
      --exp-dir "$EXP_DIR"
fi
