#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
d=reports/v6_fresh_history_query_seed101
script=scripts/slurm/v6_history_query_seed101.sh
cfg=configs/moe_sboed/v6/fresh_history_query_seed101.yaml
methods=(moe-sboed moe-sboed-uniform)
case ${1:?submit, smoke or train} in
submit)
    mkdir "$d"
    git rev-parse HEAD > "$d/revision"
    gpu=(--parsable -A jl3lab -p gpu_p --gres=gpu:H100:1 -c 4 --mem=48G --ntasks=1)
    smoke=$(sbatch "${gpu[@]}" -t 00:20:00 -J hq-smoke -o "$d/smoke-%j.log" "$script" smoke)
    printf 'smoke\t%s\n' "$smoke" > "$d/jobs.tsv"
    train=$(sbatch "${gpu[@]}" -t 04:25:00 --array=0-1%2 --dependency="afterok:$smoke" --kill-on-invalid-dep=yes -J hq-v6 -o "$d/train-%A_%a.log" "$script" train)
    printf 'train\t%s\n' "$train" >> "$d/jobs.tsv"
    cat "$d/jobs.tsv"
    ;;
smoke|train)
    test "$(git rev-parse HEAD)" = "$(cat "$d/revision")"
    export CONDA_PREFIX="/work/jl3lab/$USER/conda/power_grid" CONDA_DEFAULT_ENV=power_grid
    export PATH="$CONDA_PREFIX/bin:$PATH" PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
    if [[ $1 == smoke ]]; then
        python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
        python - "$d" "$cfg" <<'PY'
import hashlib,sys
from pathlib import Path
import numpy as np
import yaml
from src.research.moe_sboed.v6.evaluate import ARCHIVE,sha
cfg=yaml.safe_load(Path(sys.argv[2]).read_text())
rng=np.random.default_rng(cfg["development"]["seed"])
p=Path(sys.argv[1])/"development_validation_bank.npz"
with p.open("xb") as f:
    np.savez(f,truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),
             contrasts=rng.standard_normal((1000000,4)))
assert sha(p)==cfg["development"]["bank_sha256"]
assert sha(ARCHIVE/"source/frozen_evaluator.py")==cfg["evaluation"]["scorer_sha256"]
PY
        for method in "${methods[@]}"; do
            bash run.sh --moe-sboed-v6 --config "$cfg" --method "$method" --seed 101 --smoke
        done
    else
        method=${methods[$SLURM_ARRAY_TASK_ID]}
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 15 > "$d/$method.gpu.csv" &
        sampler=$!
        trap 'kill "$sampler" 2>/dev/null || true' EXIT
        flags=(--moe-sboed-v6 --config "$cfg" --method "$method")
        bash run.sh "${flags[@]}" --seed 101 | tee "$d/$method.train"
        run=$(awk '/^OUTPUT /{print $2;exit}' "$d/$method.train")
        bash run.sh "${flags[@]}" --evaluate --development --bank-dir "$d" --allow-time-limited --run-dir "$run" | tee "$d/$method.eval"
        test -s "$run/development_evaluation.npz"
    fi
    ;;
*) exit 2;;
esac
