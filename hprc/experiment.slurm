#!/bin/bash
# Generic formal run/stage wrapper. Submit from the project root.
# sbatch hprc/experiment.slurm /absolute/source_snapshot run --config ...
# Set --output to the run's logs directory when submitting.
#SBATCH --job-name=boed-experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail
SOURCE=${1:?Provide an absolute source snapshot directory}
ENTRY=${2:?Choose run or sweep}
shift 2
[[ "$SOURCE" = /* && -d "$SOURCE" ]] || { echo "Invalid source directory" >&2; exit 2; }
cd "$SOURCE"
source hprc/environment.sh
case "$ENTRY" in
  run) exec bash run.sh "$@" ;;
  sweep) exec bash sweep_run.sh "$@" ;;
  *) echo "Unsupported entrypoint: $ENTRY" >&2; exit 2 ;;
esac
