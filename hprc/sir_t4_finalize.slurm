#!/bin/bash
#SBATCH --job-name=sir-t4-verify
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/sir-t4-verify-%j.out
set -euo pipefail
RUN_DIR=$1
python3 "$RUN_DIR/source_snapshot/tools/rerun_sir_t4.py" finalize --run-dir "$RUN_DIR"
