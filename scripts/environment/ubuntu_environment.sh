#!/usr/bin/env bash
# Source this before running the project on ubuntu-pc (RTX 5080 / CUDA 12.8).
# This configures the shell only; it does not launch an experiment.
_ubuntu_env_prefix="$HOME/miniconda3/envs/mocu_optimized"
if [[ ! -x "$_ubuntu_env_prefix/bin/python" || ! -x "$_ubuntu_env_prefix/bin/x86_64-conda-linux-gnu-g++" ]]; then
    echo 'The mocu_optimized environment and Conda C++ compiler are required.' >&2
    return 1
fi
export PATH="$_ubuntu_env_prefix/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin=$_ubuntu_env_prefix/bin/x86_64-conda-linux-gnu-g++"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
unset _ubuntu_env_prefix
