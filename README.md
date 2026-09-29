# Objective-Driven Sequential BOED

Sequential Bayesian experimental design for power-grid and SIR models, comparing
DAD, RL-sBOED, Step-DAD, Myopic, Fixed and Random. Active objectives are
EIG and cost_utility. IEEE9 supports both. Probes have continuous
duration, fixed injection bus/amplitude and consecutive three-second recording
windows; the physical state carries between probes. Simulation runs online. For T=3, durations strictly increase with at least
a 0.01-second gap.

## Current conference study

The short conference paper targets **EIG on IEEE9, IEEE14 and IEEE30**, comparing
**DAD, RL-sBOED, Step-DAD, Myopic, Fixed and Random**. SIR-ODE is a supplementary
benchmark. MoE remains available only when explicitly selected and is outside
this paper. Retired objective implementations have been removed from active source;
historical results and frozen snapshots remain unchanged.

Select a single method with `run.sh --method <name>` or the six grid methods with
`--methods dad,rl_sboed,step_dad,myopic,fixed,random`. Step-DAD requires its DAD
policy. A standalone DAD run uses random initialization; a joint Fixed+DAD run
can select Fixed initialization on validation. Match and report initialization
when comparing runs. Baseline execution and timing do not require MoE modules.

IEEE9 is implemented. IEEE14 EIG passed LabPC CPU/GPU and seven-method smoke validation; see
[IEEE14 readiness](docs/ieee14_eig.md). IEEE30 still requires its specified full
dynamic backend; listing it as a target does not make it runnable.

## Installation

```bash
conda create -n mocu_optimized python=3.10 -y
conda activate mocu_optimized
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

A CUDA GPU and compatible CUDA toolkit are required for the online grid runner.

## Cost utility in plain language

The purpose is to spend less supplementary control while satisfying a declared
joint frequency and RoCoF safety requirement. All six methods use the same rule.

1. After the last probe, update the posterior using noisy observations. Each
   parameter particle carries its own physical state; evaluator truth is excluded.
2. For each candidate control, simulate the declared contingency and control from
   those states. A particle is safe only if BOTH its minimum frequency deviation
   is at least `delta_f_nadir_hz` AND its maximum absolute RoCoF is at most
   `rocof_limit_hz_s`. The metrics span all modeled dynamic buses and the entire
   terminal control window, including initial frequency.
3. Add the posterior weights of the jointly safe particles. Choose the smallest
   continuous `u_ctrl` whose safe probability is at least `posterior_coverage`.
4. Score the experiment with `cost_utility = -u_ctrl / u_max`. Average that score
   across complete episodes. Higher is better; the best possible score is zero.

Example with `u_max = 0.50` and coverage `0.90`:

| Selected control | Joint posterior safety | Utility | Admissible? |
|---:|---:|---:|---|
| 0.20 | 0.93 | -0.40 | Yes |
| 0.30 | 0.96 | -0.60 | Yes, but higher effort |
| 0.10 | 0.85 | Not eligible | No |

These are illustrative outcomes, not measured results. The minimum admissible
control is selected within each history; policy comparisons average across
histories. Safety is a constraint, not an arbitrary weighted penalty. The score
is normalized effort, not dollars. Because the step duration and power base are
fixed, effort is proportional to delivered energy. No oracle cost is subtracted.
This retains the former minimum-safe-control decision rule, with an explicit
normalized utility; renaming it is not a claim of a new mathematical objective.

Infeasible particles retain their posterior mass. If no bounded control can meet
coverage, the run fails explicitly and saves failure metadata. It never clips to
maximum control, drops particles, or resamples systems. A failed run cannot be
ranked as a successful low-cost method. Numerical training also stops on an
infeasible counterfactual; no hidden finite failure penalty is used.

The solver checks sampled monotonicity, brackets a transition, bisects to the
configured tolerance, and directly verifies the selected action. It rejects
observed nonmonotone safe sets; it is not a global safety certificate between
samples. Finite particles and numerical integration also introduce error.

Reports include utility, raw control, separate frequency/RoCoF safety rates,
joint safety, and the true-system minimum-control diagnostic. Posterior coverage
is not a guaranteed held-out safety rate. IEEE9 currently retains coverage 0.90,
frequency-deviation threshold -0.20 Hz, and RoCoF limit 22 Hz/s; these are study
settings, not asserted engineering standards. Control starts immediately, with
zero decision latency, and stays constant for the configured control window.
Probe-period safety is NOT enforced by this implementation. IEEE14 cost utility
and the full IEEE30 backend remain unavailable pending validation/implementation.

DAD, Fixed, and Step-DAD optimize this nonsmooth utility using paired antithetic
parameter perturbations (a declared numerical adaptation). RL-sBOED uses
terminal-utility-equivalent rewards; Myopic maximizes expected next-stage utility;
Random samples feasible durations. EIG and its trainers remain separate.

## Running experiments

New continuous-grid EIG runs default to **1,024 contrasts** for training, validation and evaluation (sPCE ceiling `log(1025) = 6.93245` nats). Override with `--contrasts`; planner particles are a separate setting. Existing runs retain their saved budgets.

Use the maintained shell entrypoints. This example is a workability smoke test:

```bash
bash run.sh --config configs/ieee9_cost_utility.yaml --objective cost_utility --coverage 0.9 \
  --T 3 --N_obs 0 --noise_sigma 0.005 --seed 101 --eval-seeds 1001 \
  --smoke
```

Use `--objective eig --config configs/ieee9_eig.yaml` for EIG.
Only `eig` and `cost_utility` are accepted objective names. Output directories
must be new. For a sweep:

```bash
bash sweep_run.sh --configs ieee9_cost_utility --objective cost_utility --coverage 0.9 \
  --T 3,4,5 --N_obs 0 --noise_sigma 0.005 --seed 101 --eval-seeds 1001
```

Set training, evaluation and planning budgets explicitly for performance studies;
smoke tests establish workability only. Each run saves its settings, policies,
ordered observations, evaluations and completion status under `experiments/`.
No equilibrium probe or control bank is used by the new grid protocol.

IEEE14 EIG support and its validation procedure are documented in
[IEEE14 EIG](docs/ieee14_eig.md). Check the validation receipt before full runs.
IEEE30 and IEEE14 cost_utility remain blocked in the continuous CLI. SIR remains available through
`run.sh --config configs/sir_ode_eig.yaml --experiment_type eig_based`.
Run checks with `bash scripts/check.sh` in the activated environment.
See [AGENTS.md](AGENTS.md) for authorization, method adaptations, detailed
workflows, reporting and HPRC instructions. Manuscript sources are in `documents/`.

Power-grid runs default to a scalar maximum absolute RoCoF observation (`N_obs=0`) with continuous probe duration and non-reset 3-second windows. Noise `0.005` is in Hz/s and is applied to the extracted peak. Positive `N_obs` explicitly selects frequency samples with noise in Hz. IEEE14 EIG uses an explicit objective override; IEEE30 remains unavailable.

Results are grouped by start date: `experiments/MMDDYYYY/MMDDYYYY_<time>_<objective_parameters>/`. Existing runs are listed in `experiments/INDEX.csv`.

### Optional additional evaluation seeds

Fresh training remains the default. To evaluate completed matching checkpoints
without retraining, explicitly select their run directory:

    bash run.sh --evaluate-from /absolute/path/to/completed/run --eval-seeds 1002,1003,1004,1005

Source settings are inherited and checked. Results go to a new dated directory
with checkpoint hashes and training-run provenance. Use the original completed
training run (models and source_snapshot must be present), not a lightweight
result-only copy. Different training settings require a fresh run.
This option does not resume training and is not enabled for sweeps.

## Independent MoE-sBOED for SIR ODE

Select `--method moe_sboed` explicitly for the SIR EIG benchmark. The independent
policy uses four experts and learned top-2 sparse routing at every stage. Training
starts from random weights, uses PPO on terminal information gain, and selects
the checkpoint by validation EIG only. It uses no baseline checkpoint, planner
labels, fixed first action, behavioral cloning, or forced design diversity.

```bash
bash run.sh --config configs/sir_ode_eig.yaml --T 3 --method moe_sboed --seed 101 --eval-seeds 1001,1002,1003,1004,1005
```

Implementation: `src/policies/independent_moe.py` and
`src/objectives/eig/independent_moe.py`. The SIR metric is finite-particle posterior
entropy reduction; it is distinct from online IEEE9 sPCE. This discrete policy
is not yet connected to continuous-duration IEEE9. The continuous EIG MoE implementation is separate. Cost-utility MoE is not yet
implemented; the CLI rejects that combination rather than training on EIG.


## Direct MoE policy for continuous IEEE9 EIG

Use `--method fixed,dad,moe_sboed --moe-training-mode policy_pathwise` to
compare DAD with four independent 64-by-64 experts and a history-conditioned
soft router. Both use the same history features, feasible-duration map,
full-horizon sPCE objective, pathwise gradients, Adam and validation selection.
Including Fixed offers both policies the same newly trained sequence as an
initialization candidate; all hidden expert weights are initialized afresh.
No DAD checkpoint or imitation loss is used. Omitting Fixed gives random
initialization. This experimental mode does not establish superiority.
Training histories record routing variation and expert-proposal differences.
