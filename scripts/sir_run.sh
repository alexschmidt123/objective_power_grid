#!/bin/bash
# SIR-only legacy benchmark; grid config loading is explicitly blocked.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run.sh"
set -euo pipefail

CONFIG=""
METHOD=""
SMOKE=""
FORCE=""
BANK_STRUCTURE_AUDIT=""
EXPERIMENT_TYPE="$EXPERIMENT_TYPE_DEFAULT"
EXPERIMENT_TYPE_SET=0
EXP_DIR=""
# Optional probe horizon (default 5). Do not encode T in config filenames.
T="$DEFAULT_STEP_NUMBER"
N_OBS="$DEFAULT_N_OBS"
N_OBS_SET=0
NOISE_SIGMA="$DEFAULT_NOISE_SIGMA"
NOISE_SIGMA_SET=0
SEED="$DEFAULT_SEED"
EVAL_SEEDS="$DEFAULT_EVAL_SEEDS"

usage() {
    echo "Usage: $0 --config <config.yaml> [--T <horizon>] [--N_obs <count>] [--noise_sigma <sigma>] [--seed <training-seed>] [--eval-seeds <csv>] [--experiment_type objective_based|eig_based|msc_based] [--method <methods>] [--exp-dir <path>] [--force] [--bank-structure-audit] [--smoke]" >&2
    echo "" >&2
    echo "  --method  optional; comma-separated list (default: experiment.methods in yaml)" >&2
    echo "            e.g. --method dad,random  (skips training when all selected are baselines)" >&2
    echo "  --seed    training RNG seed (default: ${DEFAULT_SEED}; sweep uses ${DEFAULT_SEEDS})" >&2
    echo "  --eval-seeds independent evaluation RNG seeds (default: ${DEFAULT_EVAL_SEEDS})" >&2
    echo "  --T       probe horizon (default: ${DEFAULT_STEP_NUMBER})" >&2
    echo "  --N_obs   IEEE trajectory samples; 0 = scalar max-ROCOF (ignored for SIR ODE)" >&2
    echo "  --noise_sigma observation noise std (IEEE default: ${DEFAULT_NOISE_SIGMA}; SIR uses YAML)" >&2
    echo "  --force   regenerate data bank if one already exists (usually unnecessary)" >&2
    echo "  --bank-structure-audit  run Myopic-trap / redundancy audit after data gen; fail if not ready" >&2
    echo "Result folders: date_time_configname_Uctrl|EIG_Tnum_NobsN_sigmaX" >&2
    echo "Examples:" >&2
    echo "  bash run.sh --config configs/ieee9_mocu.yaml" >&2
    echo "  bash run.sh --config configs/ieee9_mocu.yaml --T 8 --seed 101" >&2
    echo "  bash run.sh --config configs/sir_ode_eig.yaml" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-config|-c) CONFIG="$2"; shift 2 ;;
        --method|-method|-m) METHOD="$2"; shift 2 ;;
        -T|--T|--step-number|--step_number) T="$2"; shift 2 ;;
        --N_obs|--n-obs|--n_obs) N_OBS="$2"; N_OBS_SET=1; shift 2 ;;
        --noise_sigma|--noise-sigma) NOISE_SIGMA="$2"; NOISE_SIGMA_SET=1; shift 2 ;;
        --seed|--seeds) SEED="$2"; shift 2 ;;
        --eval-seeds|--eval_seeds) EVAL_SEEDS="$2"; shift 2 ;;
        --experiment_type|--experiment-type)
            EXPERIMENT_TYPE="$(validate_experiment_type "$2")" || exit 1
            EXPERIMENT_TYPE_SET=1
            shift 2
            ;;
        --exp-dir|--exp_dir) EXP_DIR="$2"; shift 2 ;;
        --force) FORCE="--force"; shift ;;
        --bank-structure-audit|--require-myopic-trap)
            BANK_STRUCTURE_AUDIT=1
            shift
            ;;
        --smoke) SMOKE="--smoke"; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            usage
            exit 1
            ;;
    esac
done

[[ -n "$CONFIG" ]] || { usage; exit 1; }

# Resolve objective-specific observation defaults before validating / stamping.
# A flat observation block remains supported for legacy configs and SIR. Explicit
# --N_obs / --noise_sigma always win over the selected YAML subtree.
mapfile -t OBSERVATION_DEFAULTS < <(python3 -c '
import sys, yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
etype = str(sys.argv[2]).strip().lower().replace("-", "_")
obs = dict(raw.get("observation") or {})
profile = obs.get(etype)
active = dict(profile) if isinstance(profile, dict) else obs
print(int(active.get("N_obs", 0)))
print(float(active.get("noise_sigma", 0.005)))
' "$CONFIG" "$EXPERIMENT_TYPE")
if [[ "$N_OBS_SET" -eq 0 ]]; then
    N_OBS="${OBSERVATION_DEFAULTS[0]}"
fi
if [[ "$NOISE_SIGMA_SET" -eq 0 ]]; then
    NOISE_SIGMA="${OBSERVATION_DEFAULTS[1]}"
fi

[[ "$T" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --T: $T (positive integer required)" >&2; exit 1; }
[[ "$N_OBS" =~ ^[0-9]+$ ]] || { echo "Invalid --N_obs: $N_OBS (non-negative integer required)" >&2; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid --seed: $SEED (non-negative integer required)" >&2; exit 1; }
IFS=',' read -r -a EVAL_SEED_ARR <<< "$EVAL_SEEDS"
for eval_seed in "${EVAL_SEED_ARR[@]}"; do
    [[ "$eval_seed" =~ ^[0-9]+$ ]] || { echo "Invalid evaluation seed: $eval_seed" >&2; exit 1; }
done
[[ ${#EVAL_SEED_ARR[@]} -gt 0 ]] || { echo "At least one evaluation seed is required" >&2; exit 1; }

# SIR ODE: EIG-only; design=time, observe infected count. Ignore IEEE N_obs.
IS_SIR_FLAG="$(python3 -c '
import sys, yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
name = str((raw.get("system") or {}).get("name") or "").lower().replace("-", "_")
print("1" if name in {"sir_ode", "sir"} or raw.get("sir_ode") or raw.get("sir") else "0")
' "$CONFIG")"
if [[ "$IS_SIR_FLAG" == "1" ]]; then
    if [[ "$EXPERIMENT_TYPE_SET" -eq 1 && "$EXPERIMENT_TYPE" != "eig_based" ]]; then
        echo "SIR ODE supports eig_based only (got --experiment_type=$EXPERIMENT_TYPE)" >&2
        exit 1
    fi
    EXPERIMENT_TYPE="eig_based"
    if [[ "$N_OBS_SET" -eq 1 ]]; then
        echo "[run.sh] ignoring --N_obs for SIR ODE (scalar infected-count observation)" >&2
    fi
    N_OBS=1
    if [[ "$NOISE_SIGMA_SET" -eq 0 ]]; then
        NOISE_SIGMA="$(python3 -c '
import sys, yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
obs = dict(raw.get("observation") or {})
sir = dict(raw.get("sir_ode") or raw.get("sir") or {})
print(float(obs.get("noise_sigma", sir.get("likelihood_sigma", 1.0))))
' "$CONFIG")"
    fi
    echo "[run.sh] SIR ODE mode: experiment_type=eig_based, chronological times, y=I(t) count"
fi

python3 -c 'import sys; x=float(sys.argv[1]); assert x > 0' "$NOISE_SIGMA" 2>/dev/null \
    || { echo "Invalid --noise_sigma: $NOISE_SIGMA (positive float required)" >&2; exit 1; }

METHOD_FILTER=""
if [[ -n "$METHOD" && "${METHOD,,}" != "all" ]]; then
    METHOD_FILTER="$METHOD"
fi

mapfile -t RESOLVED_METHODS < <(
    resolve_experiment_method_keys "$CONFIG" "$T" "$N_OBS" "$NOISE_SIGMA" "$METHOD_FILTER"
) || true
if [[ ${#RESOLVED_METHODS[@]} -eq 0 ]]; then
    echo "No methods resolved (check --method / experiment.methods)" >&2
    exit 1
fi

mapfile -t TRAIN_METHODS < <(
    resolve_training_method_keys "$CONFIG" "$T" "$N_OBS" "$NOISE_SIGMA" "$METHOD_FILTER"
) || true

TYPE_ARGS=(--experiment_type "$EXPERIMENT_TYPE")
T_ARGS=(-T "$T")
OBS_ARGS=(--N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA")
if [[ -z "$EXP_DIR" ]]; then
    EXP_DIR="$(python3 -m src.experiment allocate-dir --config "$CONFIG" "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}")"
fi
# Keep only the path line (tolerate any stray stderr noise).
EXP_DIR="$(printf '%s' "$EXP_DIR" | tr -d '\r' | tail -n 1 | sed 's/[[:space:]]*$//')"
EXP_ARGS=(--exp-dir "$EXP_DIR")

# Log into the result folder once the path is known (nested scripts reuse this tee).
start_run_logging "$EXP_DIR"

echo "=== run.sh (config=$CONFIG type=$EXPERIMENT_TYPE T=$T N_obs=$N_OBS noise_sigma=$NOISE_SIGMA seed=$SEED methods=${METHOD:-config}) ==="
echo "Result folder: $EXP_DIR"
echo "Evaluate: ${RESOLVED_METHODS[*]}"
if [[ ${#TRAIN_METHODS[@]} -gt 0 ]]; then
    echo "Train: ${TRAIN_METHODS[*]}"
else
    echo "Train: (skipped — eval-only methods)"
fi

# Load generate/train/eval into memory before those steps run. Bash rereads
# this file after each command; editing run.sh during training can otherwise
# resume at a stale offset and execute `--N_obs` as a command.
run_pipeline() {
    local cmd=()
    # This step is intentionally unconditional: generate-data validates and
    # reuses a complete bank, or creates it when data.generate_if_missing=true.
    # Therefore a fresh checkout needs no separate bank-generation command.
    cmd=(
        ./scripts/data_generation.sh --config "$CONFIG"
        "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}"
        --seed "$SEED"
    )
    [[ -n "$METHOD_FILTER" ]] && cmd+=(--method "$METHOD_FILTER")
    [[ -n "$FORCE" ]] && cmd+=("$FORCE")
    [[ -n "$SMOKE" ]] && cmd+=("$SMOKE")
    "${cmd[@]}"

    if [[ -n "$BANK_STRUCTURE_AUDIT" ]]; then
        echo "=== bank-structure-audit (Myopic trap / adaptive-room gate) ==="
        python3 -m src.experiment bank-structure-audit --config "$CONFIG" \
            "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}"
        local audit_rc=$?
        echo "=== bank-structure-audit done (skipping train/eval; omit flag for full runs) ==="
        echo "Done → $EXP_DIR"
        echo "EXP_DIR=$EXP_DIR"
        exit "$audit_rc"
    fi

    if [[ ${#TRAIN_METHODS[@]} -gt 0 ]]; then
        local train_csv
        train_csv="$(IFS=,; echo "${TRAIN_METHODS[*]}")"
        cmd=(
            ./scripts/training.sh --config "$CONFIG" --method "$train_csv"
            --seed "$SEED"
            "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}"
        )
        [[ -n "$SMOKE" ]] && cmd+=("$SMOKE")
        "${cmd[@]}"
    else
        echo "[run.sh] skipping training.sh (no offline trainers in selection)"
    fi

    local eval_seed eval_dest
    mkdir -p "$EXP_DIR/evaluations"
    for eval_seed in "${EVAL_SEED_ARR[@]}"; do
        echo "=== crossed evaluation train_seed=$SEED eval_seed=$eval_seed ==="
        cmd=(
            ./scripts/evaluation.sh --config "$CONFIG"
            "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}"
            --seed "$SEED" --eval-seed "$eval_seed"
        )
        [[ -n "$METHOD_FILTER" ]] && cmd+=(--method "$METHOD_FILTER")
        [[ -n "$SMOKE" ]] && cmd+=("$SMOKE")
        "${cmd[@]}"
        eval_dest="$EXP_DIR/evaluations/seed_${eval_seed}"
        mkdir -p "$eval_dest"
        rm -rf "$eval_dest/eval"
        mv "$EXP_DIR/eval" "$eval_dest/eval"
        [[ -f "$EXP_DIR/summary.md" ]] && cp "$EXP_DIR/summary.md" "$eval_dest/summary.md"
        cp "$EXP_DIR/run_config.json" "$eval_dest/run_config.json"
    done
    python3 - "$EXP_DIR" "$SEED" "$EVAL_SEEDS" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])/"evaluation_manifest.json"
p.write_text(json.dumps({"protocol":"crossed_publication_protocol_v1",
 "training_seed":int(sys.argv[2]),
 "evaluation_seeds":[int(x) for x in sys.argv[3].split(",")],
 "model_scope":"local_to_this_run_folder",
 "shared_models_across_run_folders":False},indent=2)+"\n")
PY

    echo ""
    printf "Done config=%s type=%s T=%s.\n" "$CONFIG" "$EXPERIMENT_TYPE" "$T"
    echo "EXP_DIR=$EXP_DIR"
    echo "LOG_FILE=${RUN_LOG_FILE:-}"
}

run_pipeline
