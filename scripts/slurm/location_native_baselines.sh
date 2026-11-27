#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
d=${LF15_CAMPAIGN:?Set the campaign directory}
worker=${LF15_PROJECT:?Set the worker project directory}
script=scripts/slurm/location_native_baselines.sh
case ${1:?submit, cell or report} in
submit)
    mkdir -p "$d/jobs"
    if [[ -e "$d/jobs/ids" ]]; then echo "Previous submission recorded in $d/jobs/ids" >&2; exit 1; fi
    git rev-parse HEAD > "$d/jobs/revision"
    gpu=(--parsable -A jl3lab -p gpu_p --ntasks=1 -c 4 --mem=64G --gres=gpu:H100:1 -t 12:00:00 --kill-on-invalid-dep=yes)
    for group in base score; do
        opts=(--array=0-9%2 -t 1-12:00:00)
        if [[ $group == score ]]; then opts=(--array=0-4%2 "--dependency=afterok:${LF15_SCORE_ENV_JOB:?}"); fi
        job=$(sbatch "${gpu[@]}" "${opts[@]}" -J "lf15-$group" -o "$d/logs/$group-%A_%a.log" "$script" cell "$group")
        echo "$group $job" | tee -a "$d/jobs/ids"
    done
    dependency=$(awk '{print $2}' "$d/jobs/ids" | paste -sd:)
    job=$(sbatch --parsable -A jl3lab -p batch --ntasks=1 -c 2 --mem=4G -t 00:15:00 --dependency="afterany:$dependency" -J lf15-report -o "$d/logs/report-%j.log" "$script" report)
    echo "report $job" | tee -a "$d/jobs/ids"
    ;;
cell|report)
    test "$(git rev-parse HEAD)" = "$(cat "$d/jobs/revision")"
    export CONDA_PREFIX="/work/jl3lab/$USER/conda/power_grid" CONDA_DEFAULT_ENV=power_grid
    export PATH="$CONDA_PREFIX/bin:$PATH" PYTHONPATH="$root" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
    if [[ $1 == report ]]; then
        python tools/location_baseline_summary.py "$d"
        sacct -j "$(awk '{print $2}' "$d/jobs/ids" | paste -sd,)" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocTRES -P > "$d/accounting.tsv"
        exit
    fi
    if [[ $2 == base ]]; then
        seed=$((101 + SLURM_ARRAY_TASK_ID / 2))
        if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then methods=(random fixed dad); else methods=(vsoed rl_sboed); fi
        status=0
        for method in "${methods[@]}"; do
            bash "$script" cell "$method" "$seed" || status=1
        done
        exit "$status"
    fi
    if [[ $2 == score ]]; then
        method=scorebed
        seed=$((101+SLURM_ARRAY_TASK_ID))
        export BOED_SCOREBED_PYTHON="/work/jl3lab/$USER/conda/lf15_scorebed/bin/python"
        "$BOED_SCOREBED_PYTHON" -c 'import jax; assert any(d.platform=="gpu" for d in jax.devices()), jax.devices()'
    else
        method=$2
        seed=${3:?missing seed}
    fi
    nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used --format=csv -l 30 > "$d/logs/$method-$seed-gpu.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true' EXIT
    bank="$worker/experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109/evaluation/development_validation_bank.npz"
    export BOED_TRAINING_START_MONOTONIC=$(python -c 'import time; print(time.monotonic())')
    timeout --kill-after=30s 12h bash run.sh --config configs/location_finding/location_native_baselines.yaml --location-baselines --method "$method" --seed "$seed" --bank "$bank" --experiments "$worker/experiments" --receipt "$d/$method-$seed.json"
    ;;
*) exit 2;;
esac
