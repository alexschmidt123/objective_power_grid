#!/bin/bash
# Source before importing PyCUDA, matplotlib, or torch. Never default to HOME.
# Slurm's node-local temporary directory is removed by the scheduler after a job.
boed_prepare_caches() {
    local base probe
    if [[ -n "${BOED_CACHE_ROOT:-}" ]]; then
        base="$BOED_CACHE_ROOT"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/boed-${USER:?}-${SLURM_JOB_ID}"
    else
        base="${TMPDIR:-/tmp}/boed-${USER:?}-cache"
    fi
    [[ "$base" = /* ]] || { echo "BOED_CACHE_ROOT must be absolute" >&2; return 1; }
    export XDG_CACHE_HOME="$base/xdg"
    export PYCUDA_CACHE_DIR="$base/pycuda"
    export MPLCONFIGDIR="$base/matplotlib"
    export CUDA_CACHE_PATH="$base/cuda"
    export TORCH_EXTENSIONS_DIR="$base/torch_extensions"
    export PYTHONDONTWRITEBYTECODE=1
    local directory
    for directory in "$XDG_CACHE_HOME" "$PYCUDA_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"; do
        mkdir -p "$directory" || return 1
        probe=$(mktemp "$directory/.write-check.XXXXXX") || return 1
        printf 'cache preflight\n' > "$probe" || return 1
        rm -f -- "$probe" || return 1
    done
    printf '[cache] verified writable caches under %s\n' "$base"
}
boed_prepare_caches
unset -f boed_prepare_caches
