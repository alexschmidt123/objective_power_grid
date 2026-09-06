#!/bin/bash
#SBATCH --job-name=ieee9-duration-q95
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/ieee9-duration-q95-%j.out
source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0 Python/3.10.8 CUDA/12.1.1
export PATH=/scratch/user/g.lin/venvs/objective_power_grid/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
cd /home/g.lin/objective_power_grid
python tools/test_mocu_space_audit.py
python tools/check_mocu_alignment.py
python -u tools/audit_ieee9_duration_combos.py --candidates 32 --finalists 3 --horizons 2,3,4,5 --seeds 101,202,303
