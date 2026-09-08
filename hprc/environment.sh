#!/bin/bash
# Source from a Slurm allocation; override BOED_VENV for another environment.
# Site profile scripts assume non-strict login-shell behavior.
set +eu
set +o pipefail
source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0 Python/3.10.8 CUDA/12.1.1
export PATH="${BOED_VENV:-/scratch/user/g.lin/venvs/objective_power_grid}/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
