# ubuntu-pc setup

Project: `/home/qiqi/Documents/objective_power_grid` on `ssh ubuntu-pc`.
Hardware: NVIDIA RTX 5080, 16 GB GPU memory. LabPC remains the primary source.

## Open a project shell

```bash
ssh ubuntu-pc
cd /home/qiqi/Documents/objective_power_grid
source scripts/ubuntu_environment.sh
```

The environment script selects the existing `mocu_optimized` Python environment
and its Conda C++ compiler for CUDA kernel compilation. It does not start jobs.
The installed environment uses PyTorch 2.11.0+cu128 and CUDA toolkit 12.8.

Use the normal `bash run.sh ...` / `bash sweep_run.sh ...` entrypoints after
sourcing the environment. Keep scientific settings explicit and preserve each
run's source/configuration record. The conference methods are
`dad,rl_sboed,step_dad,myopic,fixed,random`; MoE remains available when requested.

## Setup evidence

See `experiments/09182026/09182026_ubuntu_setup/` for the deployment file hashes,
GPU checks, smoke results and recorded package versions. This deployment is a
snapshot of the current LabPC working tree, including uncommitted source fixes;
the base Git commit alone does not describe it. The original project at
`~/Documents/dad_mocu_kuramoto_v4` is preserved. Historical results, simulation
banks, and the TPEC conference result collection are not part of this runtime
source deployment. Local assistant instructions stay on LabPC.

IEEE30's specified dynamic backend is not implemented. Moving the project does
not make that case runnable. IEEE14 checks establish only the documented
reduced-model numerical behavior, not publication-level experimental results.
