#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
d=${V6_STUDY_DIR:?Set an absolute study directory}
project=${V6_PROJECT_DIR:?Set the worker project directory}
bank="$project/experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109/evaluation"
script=scripts/slurm/v6_expert_roles.sh

case ${1:?submit, smoke, train, or report} in
submit)
    mkdir "$d/jobs"
    git rev-parse HEAD > "$d/jobs/revision"
    gpu=(--parsable -A jl3lab -p gpu_p --gres=gpu:H100:1 --ntasks=1 -c 4 --mem=48G --kill-on-invalid-dep=yes)
    smoke=$(sbatch "${gpu[@]}" -t 00:25:00 --dependency="afterany:${2:?provide preceding baseline job ID}" -J v6-roles-smoke -o "$d/smoke-%j.log" "$script" smoke)
    echo "smoke $smoke" | tee "$d/jobs/ids"
    train=$(sbatch "${gpu[@]}" -t 05:30:00 --array=0-3%2 --dependency="afterok:$smoke" -J v6-roles -o "$d/train-%A_%a.log" "$script" train)
    echo "train $train" | tee -a "$d/jobs/ids"
    report=$(sbatch --parsable -A jl3lab -p batch --ntasks=1 -c 1 --mem=4G -t 00:10:00 --dependency="afterany:$train" -J v6-roles-report -o "$d/report-%j.log" "$script" report)
    echo "report $report" | tee -a "$d/jobs/ids"
    ;;
smoke|train|report)
    test "$(git rev-parse HEAD)" = "$(cat "$d/jobs/revision")"
    export CONDA_PREFIX="/work/jl3lab/$USER/conda/power_grid" CONDA_DEFAULT_ENV=power_grid
    export PATH="$CONDA_PREFIX/bin:$PATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
    if [[ $1 == report ]]; then
        python -m tools.moe_sboed.v6.expert_roles_report "$d"
        sacct -j "$(awk '{print $2}' "$d/jobs/ids" | paste -sd,)" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocTRES -P > "$d/accounting.tsv"
        exit
    fi
    if [[ $1 == smoke ]]; then
        python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
        python -c 'import sys,yaml; from pathlib import Path; from src.research.moe_sboed.v6.evaluate import require_originals; require_originals(Path(sys.argv[1]),yaml.safe_load(Path("configs/moe_sboed/v6/observed_history_seed101.yaml").read_text())["evaluation"])' "$bank"
        cells=(0 1 2 3)
    else
        cells=("$SLURM_ARRAY_TASK_ID")
        nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used --format=csv -l 30 > "$d/$SLURM_ARRAY_TASK_ID.gpu.csv" &
        sampler=$!
        trap 'kill "$sampler" 2>/dev/null || true' EXIT
    fi
    architectures=(observed structured)
    methods=(moe-sboed moe-sboed-uniform)
    for i in "${cells[@]}"; do
        name=${architectures[i/2]}_history_seed101
        method=${methods[i%2]}
        flags=(--moe-sboed-v6 --config "configs/moe_sboed/v6/$name.yaml" --method "$method")
        if [[ $1 == smoke ]]; then
            bash run.sh "${flags[@]}" --seed 101 --experiments "$project/experiments" --smoke
        else
            bash run.sh "${flags[@]}" --seed 101 --experiments "$project/experiments" | tee "$d/$i.train"
            run=$(awk '/^OUTPUT /{print $2;exit}' "$d/$i.train")
            printf '%s\n' "$run" > "$d/$i.path"
            bash run.sh "${flags[@]}" --evaluate --allow-time-limited --bank-dir "$bank" --run-dir "$run"
            test -s "$run/canonical_evaluation.npz"
        fi
    done
    ;;
*) exit 2;;
esac
