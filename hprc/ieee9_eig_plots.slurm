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

RUNTIME=/scratch/user/g.lin/objective_power_grid_runtime
CAMPAIGN=ieee9_eig_pub_20260904
FIRST="$RUNTIME/experiments/09042026_025439_ieee9_eig_hprc_EIG_T3_Nobs5_sigma0p01"
ARGS=(
  python -m src.results.plots
  --stamp "$(date +%m%d%Y_%H%M%S)"
  --experiment-type eig_based
  --T 3,4,5
  --exp-dir "$FIRST"
)
for task in {0..7}; do
  manifest="$RUNTIME/manifests/${CAMPAIGN}_task${task}.path"
  test -s "$manifest"
  ARGS+=(--exp-dir "$(<"$manifest")")
done
"${ARGS[@]}"
