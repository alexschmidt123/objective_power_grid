#!/bin/bash
# Full experiment: call core scripts in order.
#
#   bash run.sh --config configs/ieee9_mocu.yaml
#   bash run.sh --config configs/ieee9_mocu.yaml --T 8
#   bash run.sh --config configs/ieee9_eig.yaml --experiment_type eig_based
#   bash run.sh --config configs/sir_ode_eig.yaml
#   bash run.sh --config configs/ieee9_mocu.yaml --method dad --force
#
# Result folder (allocated once, reused for all steps):
#   experiments/date_time_configname_Uctrl|EIG_Tnum_NobsN_sigmaX
# Full terminal history is saved as <result_folder>/logs/run_log.log
#
# Nested scripts may ``source`` this file for shared env/helpers only
# (when sourced, the main pipeline below does not run).

# --- shared env / helpers (also used when this file is sourced) ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Prefer conda env mocu_optimized without hardcoding a user home.
# If that env is already active, leave PATH alone; otherwise prepend a
# standard install location so both workstations work without `conda activate`.
_prepend_mocu_optimized() {
    local bin candidates=()
    if [[ "${CONDA_DEFAULT_ENV:-}" == "mocu_optimized" && -x "${CONDA_PREFIX:-}/bin/python3" ]]; then
        return 0
    fi
    candidates+=(
        "${HOME}/miniconda3/envs/mocu_optimized/bin"
        "${HOME}/anaconda3/envs/mocu_optimized/bin"
        "${HOME}/miniforge3/envs/mocu_optimized/bin"
        "${HOME}/mambaforge/envs/mocu_optimized/bin"
    )
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        candidates+=("$(dirname "${CONDA_PREFIX}")/mocu_optimized/bin")
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
        candidates+=("$(dirname "$(dirname "${CONDA_EXE}")")/envs/mocu_optimized/bin")
    fi
    for bin in "${candidates[@]}"; do
        if [[ -n "$bin" && -x "${bin}/python3" ]]; then
            export PATH="${bin}:${PATH}"
            return 0
        fi
    done
    return 0
}
_prepend_mocu_optimized
unset -f _prepend_mocu_optimized
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT_TYPE_DEFAULT="objective_based"
DEFAULT_STEP_NUMBER=3
DEFAULT_N_OBS=5
DEFAULT_NOISE_SIGMA=0.005
DEFAULT_SEED=101
# Publication training RNGs (sweep cartesian axis). Bank θ uses yaml
# train_seed/test_seed; this is the policy-training seed only.
DEFAULT_SEEDS="101,202,303"
DEFAULT_EVAL_SEEDS="1001,1002,1003,1004,1005"

validate_experiment_type() {
    local t="${1,,}"
    t="${t//-/_}"
    case "$t" in
        objective_based|eig_based|msc_based)
            echo "$t"
            return 0
            ;;
        *)
            echo "Invalid --experiment_type: $1 (allowed: objective_based|eig_based|msc_based)" >&2
            return 1
            ;;
    esac
}

METHODS_HELP="dad, rl_sboed, moe_sboed, myopic, fixed, random, matched_dense, step_dad (comma-separated for multiple)"

# Resolve evaluate/train method keys via Python (honours config + comma lists).
resolve_experiment_method_keys() {
    local config="$1"
    local t="$2"
    local n_obs="$3"
    local sigma="$4"
    local method_filter="${5-}"
    python3 -c '
import sys
from src.experiment import load_experiment_config
from src.objectives.mocu.context import methods_from_args

cfg = load_experiment_config(
    sys.argv[1],
    step_number=int(sys.argv[2]),
    n_obs=int(sys.argv[3]),
    noise_sigma=float(sys.argv[4]),
)
method_filter = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
for key in methods_from_args(cfg, method_filter):
    print(key)
' "$config" "$t" "$n_obs" "$sigma" "$method_filter"
}

resolve_training_method_keys() {
    local config="$1"
    local t="$2"
    local n_obs="$3"
    local sigma="$4"
    local method_filter="${5-}"
    python3 -c '
import sys
from src.experiment import load_experiment_config
from src.objectives.mocu.context import methods_from_args, training_method_keys

cfg = load_experiment_config(
    sys.argv[1],
    step_number=int(sys.argv[2]),
    n_obs=int(sys.argv[3]),
    noise_sigma=float(sys.argv[4]),
)
method_filter = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
eval_keys = methods_from_args(cfg, method_filter)
for key in training_method_keys(eval_keys):
    print(key)
' "$config" "$t" "$n_obs" "$sigma" "$method_filter"
}

# Tee all stdout/stderr to <result_dir>/logs/run_log.log (idempotent for nested scripts).
start_run_logging() {
    if [[ -n "${RUN_LOG_ACTIVE:-}" ]]; then
        return 0
    fi
    local result_dir="${1:-}"
    if [[ -z "$result_dir" ]]; then
        echo "start_run_logging: result directory required" >&2
        return 1
    fi
    result_dir="$(printf '%s' "$result_dir" | tr -d '\r' | sed 's/[[:space:]]*$//')"
    mkdir -p "${result_dir%/}/logs"
    export RUN_LOG_FILE="${result_dir%/}/logs/run_log.log"
    : > "${RUN_LOG_FILE}"
    export RUN_LOG_ACTIVE=1
    exec > >(tee -a "${RUN_LOG_FILE}") 2>&1
    echo "Log file: ${RUN_LOG_FILE}"
}

# Sourced by sweep_run.sh / scripts/*.sh — setup only.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

set -euo pipefail
exec python3 -m src.online_cli "$@"
