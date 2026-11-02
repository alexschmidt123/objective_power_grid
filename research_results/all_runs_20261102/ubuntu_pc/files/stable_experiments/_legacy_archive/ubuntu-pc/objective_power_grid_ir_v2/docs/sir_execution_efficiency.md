# On-the-fly SIR ODE execution study

The Ubuntu seed-101 development campaign uses exactly ten main methods and
retains one million total primary training episodes per trained method. The
internal 500k SIR base is shared by the two residual variants, which each train
500k additional episodes. It is not an additional baseline. Native objectives,
policy architectures, optimizer settings and batch sizes are unchanged.

Every newly simulated episode samples fresh infection/recovery rates and
Gaussian observation noise and integrates the carried SIR state at chosen
times. Native replay may reuse generated episodes. No precomputed epidemic-path
bank supplies observations. The common fixed EIG contrast bank is evaluation
randomness, not a training simulator. See the [benchmark clarification](research_records/20261017_sir_execution/on_the_fly_sir.md).

## Execution changes

- `sir_ode.execution_backend: fused_rk4` optionally fuses the original float64
  adaptive RK4 calculation. Its detached per-particle subdivision counts,
  minimum 512 steps, zero-length padding, carried state and likelihood remain
  unchanged. The backward differentiates the discrete RK4 operations, rather
  than approximating a continuous adjoint. Higher-order derivatives are
  explicitly unsupported; the native pathwise trainers use first derivatives.
  The existing eager backend remains the default. CPU execution falls back to
  the original solver. An intermediate compiled-block backend is retained for
  reproducible execution comparisons, not as a new MoE method.
- Optional GPU diagnostic policy rollouts preserve the CPU-generated paired
  truth/noise/contrast draws. The common likelihood scorer already used GPU;
  this change does not introduce a new estimator.
- Diagnostic validation uses 64 systems/L=1024 every 125k episodes, instead of
  128/L=16384. It may saturate and cannot select a checkpoint or establish EIG
  convergence. Final-budget checkpoint selection and full development scoring
  (2048 paired systems, L=16384, seed1001) are unchanged. Estimator resolution
  must still be checked on developed checkpoints before confirmation.

## Measured checks

Hardware: one RTX5080 16GB; Python3.10/PyTorch2.11+cu128; separate Python3.11
JAX0.7.2/CUDA12 ScoreBED environment. No full-training runtime has yet been
measured by the bounded probe.

|T=5, batch32, L=1024 probe|Eager|Warm fused|
|---|---:|---:|
|Forward/backward seconds|3.37845|0.40884|
|Projected optimization hours per 1M episodes|29.33|3.55|
|Peak CUDA allocation, bytes|352477184|34631168|

This is an approximately 8.3x execution speedup for the test policy, not a claim
that every native method is 8.3x faster. Compilation and validation are separate
costs. Initial campaign planning is36--72h, with a72h campaign cap and24h
optimization cap per phase. Incomplete capped runs must be labelled partial or
budget-limited, never completed one-million-episode results.

The paired full-rollout action/observation/sPCE/policy-gradient difference was
at most7.1e-14 in the bounded test. State/action/parameter gradients are tested
at rtol1e-7/atol1e-8; complete rollout gradients at rtol1e-6/atol1e-7.
The focused suite passed35 tests, including random carried-state VJPs and
finite-difference gradcheck. All ten native smoke runs passed. GPU validation
scores matched the fixed-draw reference within1.42e-14. ScoreBED's native GPU
ODE matched the saved independent oracle within1.14e-13 for means and9.42e-14
for time gradients. These checks establish implementation consistency, not
EIG superiority or across-seed stability.

The [raw probe](research_records/20261017_sir_execution/probe.json),
[protocol](research_records/20261017_sir_execution/journal_protocol.md), and
other small JSON records are retained. Historical checkpoints are untouched.

## Reproduction

Use a fresh dated campaign directory; the preparation script refuses to
overwrite an existing one. Activate the compatible Torch environment and
install native dependencies including pytest/joblib. Install the separate
ScoreBED environment using its upstream requirements. All full training enters
through the root launcher.

```bash
python -m tools.probe_sir_fused
python -m tools.check_sir_execution_validation
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv_scorebed/bin/python -m tools.check_sir_scorebed_gpu
python -m pytest -q -rs tools/tests/test_sir_execution.py tools/tests/test_journal_validation.py tools/tests/test_sir_offline.py tools/tests/test_sir_residual_base.py tools/tests/test_fixed_budget_selection.py tools/tests/test_scorebed_validation_metadata.py > /tmp/objective_sir_fused_tests.log 2>&1
python -m tools.prepare_ubuntu_sir_campaign
C=experiments/10162026/10162026_ubuntu_sir_t5_seed101
bash run.sh --config "$C/sir_1m.yaml" --journal --smoke --seed 101 --eval-seeds 1001 --eig-only --methods random,fixed,dad,rl_sboed --output "$C/readiness/native_smoke"
bash run.sh --config "$C/sir_1m.yaml" --journal --smoke --seed 101 --eval-seeds 1001 --eig-only --methods vsoed,scorebed,moe,switchmoe,sum_single --output "$C/readiness/native_remaining"
python -m tools.run_ubuntu_sir_campaign
```

The controller verifies readiness, tests both residual variants, seals source
and configuration hashes, serializes jobs, verifies the frozen shared base,
and records commands, failures, episode counts, costs and checkpoint hashes.
It never retries partial attempts automatically. The main results CSV has
exactly ten methods; the internal base appears only in phase accounting.

Read campaign `status.json`, `registry.json`, `sir_t5_results.csv`,
`phase_costs.csv`, `completion/*.json` and `logs/*.log` for actual progress.
Seed1001 is development evaluation;8831001 remains untouched. Independent
training seeds, estimator-resolution checks, paired and training-seed uncertainty,
baseline adaptation disclosures, expert diagnostics and full cost accounting
remain necessary for a defensible journal submission. Publication cannot be
guaranteed by runtime improvements or one seed.
