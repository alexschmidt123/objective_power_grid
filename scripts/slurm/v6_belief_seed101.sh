#!/usr/bin/env bash
# Run from the Sapelo2 main checkout. Usage: bash scripts/slurm/v6_belief_seed101.sh submit
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
script=scripts/slurm/v6_belief_seed101.sh
d=reports/v6_belief_seed101
cfg=configs/moe_sboed/v6/belief_history_spce_seed101.yaml
methods=(moe-sboed moe-sboed-uniform moe-sboed-e1 moe-sboed-e2 moe-sboed-e3 moe-sboed-e4)
case ${1:?expected submit, smoke, train or collect} in
submit)
    mkdir "$d"
    git rev-parse HEAD > "$d/revision"
    gpu=(--parsable -A jl3lab -p gpu_p --gres=gpu:H100:1 -c 4 --mem=32G --ntasks=1)
    smoke=$(sbatch "${gpu[@]}" -t 00:20:00 -J belief-smoke -o "$d/smoke-%j.log" "$script" smoke)
    printf 'smoke\t%s\n' "$smoke" > "$d/job_ids.tsv"
    study=$(sbatch "${gpu[@]}" -t 04:20:00 --array=0-5%2 --dependency="afterok:$smoke" --kill-on-invalid-dep=yes -J belief-v6 -o "$d/train-%A_%a.log" "$script" train)
    printf 'study\t%s\n' "$study" >> "$d/job_ids.tsv"
    collect=$(sbatch --parsable -A jl3lab -p batch -c 1 --mem=2G --ntasks=1 -t 00:30:00 --dependency="afterany:$study" -J belief-return -o "$d/collect-%j.log" "$script" collect)
    printf 'collect\t%s\n' "$collect" >> "$d/job_ids.tsv"
    cat "$d/job_ids.tsv"
    ;;
smoke|train)
    test "$(git rev-parse HEAD)" = "$(cat "$d/revision")"
    export CONDA_PREFIX="/work/jl3lab/$USER/conda/power_grid" CONDA_DEFAULT_ENV=power_grid
    export PATH="$CONDA_PREFIX/bin:$PATH" PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
    if [[ $1 == smoke ]]; then
        python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
        python -c 'import json; from src.research.moe_sboed.v6.evaluate import ARCHIVE, PROTOCOL, require_originals; require_originals(ARCHIVE / "evaluation", json.loads(PROTOCOL.read_text())["evaluation"])'
        for method in "${methods[@]}"; do
            bash run.sh --moe-sboed-v6 --config "$cfg" --method "$method" --seed 101 --smoke
        done
    else
        method=${methods[$SLURM_ARRAY_TASK_ID]}
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 10 > "$d/$method.gpu.csv" &
        sampler=$!
        trap 'kill "$sampler" 2>/dev/null || true' EXIT
        flags=(--moe-sboed-v6 --config "$cfg" --method "$method")
        bash run.sh "${flags[@]}" --seed 101 | tee "$d/$method.train"
        run=$(awk '/^OUTPUT /{print $2;exit}' "$d/$method.train")
        bash run.sh "${flags[@]}" --evaluate --allow-time-limited --run-dir "$run" | tee "$d/$method.eval"
        test -s "$run/canonical_evaluation.npz"
    fi
    ;;
collect)
    out="/scratch/$USER/v6_belief_seed101"
    mkdir -p "$out"
    for method in "${methods[@]}"; do
        log="$d/$method.train"
        run=""
        if [[ -f $log ]]; then run=$(awk '/^OUTPUT /{print $2;exit}' "$log"); fi
        state=incomplete
        if [[ -n $run && -s $run/canonical_evaluation.npz ]]; then state=scored; fi
        printf '%s\t%s\t%s\n' "$method" "$state" "$run"
    done > "$d/final_status.tsv"
    jobs=$(awk '{print $2}' "$d/job_ids.tsv" | paste -sd, -)
    sacct -X -j "$jobs" --format=JobID,State,Elapsed,ExitCode,AllocTRES%80 > "$d/slurm_accounting.txt"
    shopt -s nullglob
    runs=(experiments/*-belief_history_spce_seed101{,_smoke}_T30-moe-sboed*)
    tar czf "$out/experiments.tar.gz.partial" "$d" "${runs[@]}"
    mv "$out/experiments.tar.gz.partial" "$out/experiments.tar.gz"
    (cd "$out" && sha256sum experiments.tar.gz > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS)
    ;;
*) exit 2 ;;
esac
