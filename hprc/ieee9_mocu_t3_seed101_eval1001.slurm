#!/bin/bash
#SBATCH --job-name=ieee9-mocu-t3-s101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/user/g.lin/objective_power_grid_runtime/slurm/%x-%j.out
source /etc/profile
set -euo pipefail
module purge
module load GCCcore/12.2.0 Python/3.10.8 CUDA/12.1.1
export PATH=/scratch/user/g.lin/venvs/objective_power_grid/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
EXP_DIR=$1
cd "$EXP_DIR/source_snapshot"
export PYTHONPATH="$PWD"
python - <<'PY'
import yaml
c=yaml.safe_load(open('configs/ieee9_mocu.yaml'))
assert c['swing_equation']['probe_durations']==[.28,.32,.37,.40,.47,.62]
assert c['control']['robust_rule']=='quantile' and c['control']['alpha']==.05
assert c['training']['objective_based']['undercontrol_penalty']==20
assert c['training']['objective_based']['violation_penalty']==0
PY
bash run.sh --config configs/ieee9_mocu.yaml --experiment_type objective_based \
  --method dad,rl_sboed,step_dad,myopic,fixed,random --T 3 --N_obs 5 \
  --noise_sigma 0.005 --seed 101 --eval-seeds 1001 --exp-dir "$EXP_DIR"
python - "$EXP_DIR" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);d=json.loads((p/'evaluation_manifest.json').read_text())
assert d['training_seed']==101 and d['evaluation_seeds']==[1001]
assert (p/'evaluations/seed_1001/eval/summary.csv').is_file()
print('REQUESTED_RUN_COMPLETE',p,flush=True)
PY
