#!/bin/bash
#SBATCH --job-name=ieee9-master-search
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/%x-%j.out
source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0 Python/3.10.8 CUDA/12.1.1
export PATH=/scratch/user/g.lin/venvs/objective_power_grid/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
STAGE=$1
CAMPAIGN=$2
cd "$CAMPAIGN/source_snapshot"
if [[ "$STAGE" == prepare ]]; then
    python tools/test_mocu_correctness.py
    python tools/test_mocu_space_audit.py
    python tools/test_audit_validation.py
fi
python -u tools/audit_ieee9_master_search.py "$STAGE" --campaign "$CAMPAIGN" --previous "$CAMPAIGN/previous_campaign.json" --index "${SLURM_ARRAY_TASK_ID:-0}"
