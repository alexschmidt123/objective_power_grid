#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
study=${V6_STUDY_DIR:?}
project=${V6_PROJECT_DIR:?}
script=scripts/slurm/v6_location_precision.sh
case ${1:?submit, smoke, train, select, score, report} in
submit)
    mkdir "$study/jobs"
    git rev-parse HEAD > "$study/jobs/revision"
    gpu=(--parsable -A jl3lab -p gpu_p --ntasks=1 -c 4 --mem=48G --gres=gpu:H100:1 --kill-on-invalid-dep=yes)
    cpu=(--parsable -A jl3lab -p batch --ntasks=1 -c 1 --mem=4G -t 00:10:00 --kill-on-invalid-dep=yes)
    submit() {
        local name=$1
        shift
        job=$(sbatch "$@" -J "v6p-$name" -o "$study/$name-%A_%a.log" "$script" "$name")
        echo "$name $job" | tee -a "$study/jobs/ids"
    }
    submit smoke "${gpu[@]}" -t 00:20:00
    submit train "${gpu[@]}" -t 04:30:00 --array=0-3%4 --dependency="afterok:$job"
    submit select "${cpu[@]}" --dependency="afterok:$job"
    submit score "${gpu[@]}" -t 00:25:00 --array=0-1 --dependency="afterok:$job"
    submit report "${cpu[@]}" --dependency="afterany:$job"
    ;;
smoke|train|select|score|report)
    test "$(git rev-parse HEAD)" = "$(cat "$study/jobs/revision")"
    export CONDA_PREFIX="/work/jl3lab/$USER/conda/power_grid" CONDA_DEFAULT_ENV=power_grid
    export PATH="$CONDA_PREFIX/bin:$PATH" PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
    canonical="$project/experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109/evaluation"
    development="$project/experiments/location_finding/moe_sboed_v5_history_20261119"
    if [[ $1 == select ]]; then
        python -m tools.moe_sboed.v6.precision_study_report select "$study"
        exit
    elif [[ $1 == report ]]; then
        sacct -X -j "$(awk '{print $2}' "$study/jobs/ids" | paste -sd,)" --format=JobID,State,Elapsed,ExitCode,AllocTRES -P > "$study/accounting.tsv"
        python -m tools.moe_sboed.v6.precision_study_report result "$study"
        exit
    fi
    configs=(observed_precision_seed101 relative_precision_seed101)
    methods=(moe-sboed moe-sboed-uniform)
    if [[ $1 == smoke ]]; then
        python -c 'import torch; assert torch.cuda.is_available()'
        cells=(0 1 2 3)
    elif [[ $1 == score ]]; then
        base=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["learned_index"])' "$study/selection.json")
        cells=("$((base+SLURM_ARRAY_TASK_ID))")
    else
        cells=("$SLURM_ARRAY_TASK_ID")
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > "$study/$SLURM_ARRAY_TASK_ID.gpu.csv" &
        sampler=$!
        trap 'kill "$sampler" 2>/dev/null || true' EXIT
    fi
    for i in "${cells[@]}"; do
        flags=(--moe-sboed-v6 --config "configs/moe_sboed/v6/${configs[i/2]}.yaml" --method "${methods[i%2]}")
        if [[ $1 == smoke ]]; then
            bash run.sh "${flags[@]}" --seed 101 --experiments "$project/experiments" --smoke
        elif [[ $1 == train ]]; then
            bash run.sh "${flags[@]}" --seed 101 --experiments "$project/experiments" | tee "$study/$i.train"
            run=$(awk '/^OUTPUT /{print $2;exit}' "$study/$i.train")
            printf '%s\n' "$run" > "$study/$i.path"
            bash run.sh "${flags[@]}" --evaluate --development --bank-dir "$development" --allow-time-limited --run-dir "$run"
        else
            run=$(cat "$study/$i.path")
            bash run.sh "${flags[@]}" --evaluate --bank-dir "$canonical" --allow-time-limited --run-dir "$run"
        fi
    done
    ;;
*) exit 2;;
esac
