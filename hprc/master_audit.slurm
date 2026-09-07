#!/bin/bash
# Immutable IEEE9 master-bank campaign stage; submit via scripts/submit_audit.sh.
#SBATCH --job-name=boed-master-audit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
STAGE=${1:?Provide prepare, fresh, confirm, or finalize}
CAMPAIGN=${2:?Provide the campaign directory}
cd "$CAMPAIGN/source_snapshot"
source hprc/environment.sh
if [[ "$STAGE" == prepare ]]; then
    bash scripts/check.sh
fi
ARGS=(master "$STAGE" --campaign "$CAMPAIGN" --index "${SLURM_ARRAY_TASK_ID:-0}")
[[ ! -f "$CAMPAIGN/previous_campaign.json" ]] || ARGS+=(--previous "$CAMPAIGN/previous_campaign.json")
exec bash scripts/audit.sh "${ARGS[@]}"
