#!/bin/bash
#SBATCH --job-name=sir-t4-complete
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/sir-t4-complete-%A-%a.out
source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0 Python/3.10.8 CUDA/12.1.1
export PATH=/scratch/user/g.lin/venvs/objective_power_grid/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
RUN_DIR=$1
EVAL_SEED=$((1001+SLURM_ARRAY_TASK_ID))
python "$RUN_DIR/source_snapshot/tools/rerun_sir_t4.py" run --run-dir "$RUN_DIR" --seed "$EVAL_SEED"
