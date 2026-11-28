#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
study=${V6_STUDY_DIR:?}
project=${V6_PROJECT_DIR:?}
previous=${V6_REFERENCE_DIR:?}
cfg=configs/moe_sboed/v6/observed_history_k8000_seed101.yaml
reference_cfg=configs/moe_sboed/v6/observed_history_seed101.yaml
development="$project/experiments/location_finding/moe_sboed_v5_history_20261119"
canonical="$project/experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109/evaluation"
script=scripts/slurm/v6_homogeneous_contrasts.sh
case ${1:?submit, smoke, train or report} in
submit)
    mkdir "$study/k8000"
    git rev-parse HEAD > "$study/k8000/revision"
    gpu=(--parsable -A jl3lab -p gpu_p --ntasks=1 -c 4 --mem=48G --gres=gpu:H100:1 --kill-on-invalid-dep=yes)
    smoke=$(sbatch "${gpu[@]}" -J v6-k8-smoke -t 00:15:00 -o "$study/k8000/smoke-%j.log" "$script" smoke)
    echo "smoke $smoke" | tee "$study/k8000/jobs"
    diagnostic_job=$(cat "$study/diagnostic.job")
    train=$(sbatch "${gpu[@]}" -J v6-k8-train -t 04:45:00 --array=0-1 --dependency="afterok:$smoke:$diagnostic_job" -o "$study/k8000/train-%A_%a.log" "$script" train)
    echo "train $train" | tee -a "$study/k8000/jobs"
    report=$(sbatch --parsable -A jl3lab -p batch --ntasks=1 -c 1 --mem=4G -t 00:10:00 -J v6-k8-report --dependency="afterany:$train" -o "$study/k8000/report-%j.log" "$script" report)
    echo "report $report" | tee -a "$study/k8000/jobs"
    ;;
smoke|train|report)
    d="$study/k8000"
    test "$(git rev-parse HEAD)" = "$(cat "$d/revision")"
    export CONDA_PREFIX="/work/jl3lab/$USER/conda/power_grid" CONDA_DEFAULT_ENV=power_grid
    export PATH="$CONDA_PREFIX/bin:$PATH" PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
    if [[ $1 == report ]]; then
        python -m tools.moe_sboed.v6.contrast_report "$d" "$previous"
        sacct -j "$(awk '{print $2}' "$d/jobs" | paste -sd,)" --format=JobID,State,Elapsed,TotalCPU,MaxRSS,AllocTRES -P > "$d/accounting.tsv"
        exit
    fi
    methods=(moe-sboed moe-sboed-uniform)
    if [[ $1 == smoke ]]; then
        python -c 'import torch; assert torch.cuda.is_available()'
        for method in "${methods[@]}"; do
            bash run.sh --config "$cfg" --moe-sboed-v6 --method "$method" --seed 101 --experiments "$project/experiments" --smoke
        done
        # Exercise the actual contrast tensor size and compiled backward on one production microbatch.
        python - "$cfg" <<'PY'
import sys, torch, yaml
from pathlib import Path
from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.policy import build_policy
from src.research.moe_sboed.v6.objective import cumulative_likelihood, scores
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
bench = LocationFinding(cfg)
torch.manual_seed(cfg["seed"])
model = build_policy("moe-sboed", cfg, bench).cuda()
micro, k = (cfg["training"][s] for s in ("microbatch_size","contrasts"))
def draw(*size): return torch.randn(*size, device="cuda", dtype=torch.float64)
value = scores(bench, model, draw(micro,4), draw(micro,k,4), draw(micro,cfg["horizon"]), torch.compile(cumulative_likelihood,fullgraph=True)).mean()
value.backward()
assert torch.isfinite(value) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print("production microbatch smoke",value.item(),torch.cuda.max_memory_allocated(),flush=True)
PY
        exit
    fi
    i=$SLURM_ARRAY_TASK_ID
    method=${methods[i]}
    old=$(cat "$previous/$i.path")
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > "$d/$i.gpu.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true' EXIT
    bash run.sh --config "$cfg" --moe-sboed-v6 --method "$method" --seed 101 --experiments "$project/experiments" | tee "$d/$i.train"
    run=$(awk '/^OUTPUT /{print $2;exit}' "$d/$i.train")
    printf '%s\n' "$run" > "$d/$i.path"
    args=(--moe-sboed-v6 --method "$method" --evaluate --allow-time-limited)
    # Put supplemental scores for the earlier terminal model in this study.
    ref="$d/reference-$i"
    mkdir "$ref"
    for f in checkpoint.pt status.json budget.json; do ln -s "$old/$f" "$ref/$f"; done
    bash run.sh --config "$reference_cfg" "${args[@]}" --development --bank-dir "$development" --run-dir "$ref"
    bash run.sh --config "$cfg" "${args[@]}" --development --bank-dir "$development" --run-dir "$run"
    bash run.sh --config "$cfg" "${args[@]}" --bank-dir "$canonical" --run-dir "$run"
    ;;
*) exit 2;;
esac
