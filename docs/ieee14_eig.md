# IEEE14 EIG readiness and model scope

## Declared model

This experiment uses the existing five-machine reduced swing model at physical
buses 1, 2, 3, 6 and 8. It is not a full-order IEEE14 AC transient-stability model.
Twenty branch reactances and connectivity are checked against the local MATPOWER
case14 reference. The lossless graph uses 1/x weights, unit voltage magnitudes
and unit transformer taps; line resistance, shunt charging, AC power-flow
operating angles, exciters and governors are omitted. Linear Kron elimination
provides the retained coupling and physical-injection map, followed by nonlinear
sine coupling in the retained swing equations.

The configured machinewise M/K intervals and damping values are inherited study
assumptions. The network reference is power-flow data and does not establish
these dynamic priors as measured parameters. The initial deviation equilibrium
has zero angles, speeds and mechanical-injection deviations. The simulated
probe remains the existing smooth Hann pulse, amplitude .05 pu at bus 1.
No dynamics or priors were retuned to favor any method.

There are ten unknown parameters and ten carried states. A T=3 scalar observation
sequence does not generally identify all ten parameters uniquely; EIG measures
uncertainty reduction, not guaranteed full identification.

## Matched IEEE9/IEEE14 experiment protocol

IEEE14 changes the physical network and its machine-specific parameters only.
Keep the same policy architectures, objective, optimizer settings, seed protocol,
observation model, probe constraints, and Monte Carlo budgets as the established
IEEE9 signed endpoint-RoCoF experiment. Do not tune a separate IEEE14 protocol
to favor MoE. T is the requested horizon; the current matched campaign covers T=3,4,5.

| Setting | Both systems |
|---|---|
| Observation | One signed endpoint RoCoF value at bus 1 |
| Window / frequency sampling / ODE step | 3.5 s / 40 Hz / 1/640 s |
| Measurement noise | 0.005 Hz/s |
| Probe | Bus 1, amplitude 0.05 pu, same Hann waveform |
| Durations | Continuous [0.2,3] s, increasing, minimum gap 0.01 s |
| Dynamics across probes | No reset; carry all machine states |
| Train / evaluation seeds | 101 / 1001 for the current single-seed comparison |
| Updates / batch / learning rate | 2000 / 32 / 0.001 (existing method-specific optimizer rules retained) |
| EIG contrasts / planner particles | 128 / 128 |
| Validation / evaluation systems | 128 / 128; validate every 100 updates |
| Myopic search | 16 candidates, 3 rounds, 32 fantasies |
| Step-DAD | 16 refinement updates |
| RL-sBOED | 10 critics, 5 updates per batch; existing REDQ optimizer settings |
| Direct MoE | Same four 64-by-64 experts and soft router; policy_pathwise |
| Initialization | Same validation-selected random/fresh Fixed sequence procedure |

IEEE9 has three retained machines (six M/K parameters); IEEE14 has five
(ten M/K parameters). Existing machine-specific M/K bounds and damping remain
those of each physical model. Thus this is a matched-protocol comparison across
two systems, not a claim that topology alone is the only mathematical difference.
The inactive control configuration is not used by EIG.

Smoke tests and sensitivity diagnostics use explicitly reduced or varied budgets;
they do not redefine the matched performance-run protocol.

## Running EIG

Reuse the canonical physical configuration and explicitly select EIG. The file
name does not activate MOCU when --objective eig is supplied. No extra YAML is
required. IEEE14 MSC/MOCU and IEEE30 remain inactive in the continuous CLI.

~~~bash
bash run.sh --config configs/ieee14_mocu.yaml --objective eig \
  --method fixed,dad,rl_sboed,step_dad,myopic,random,moe_sboed \
  --moe-training-mode policy_pathwise --T 3 --N_obs 0 \
  --observation-kind endpoint_rocof --window 3.5 --noise_sigma .005 \
  --seed 101 --eval-seeds 1001 --updates 2000 --batch-size 32 \
  --learning-rate .001 --contrasts 128 --planner-particles 128 \
  --validation-systems 128 --validate-every 100 --eval-systems 128 \
  --duration-min .2 --duration-max 3 --min-duration-separation .01 \
  --bus 1 --amplitude .05 --search-candidates 16 --search-rounds 3 \
  --fantasies 32 --refinement-updates 16 --redq-critics 10 --redq-updates-per-batch 5
~~~

This is a launch example, not authorization for full training. First use --smoke
for workability or --estimate-only for bounded timing. Three increasing real
durations in [.2,3] with .01 separation, stage times 0/3.5/7 seconds, and signed
[f(W)-f(W-.025)]/.025 observations match the IEEE9 protocol.
The direct MoE uses four experts and the same full-horizon pathwise EIG trainer
as DAD. Shared freshly trained Fixed initialization is explicit in a joint run.
For later MoE-only training, --fixed-initialization-from requires a completed
matching IEEE14 source, never an IEEE9 checkpoint.

## Validation

CPU tests independently reconstruct the reduced network from the reference,
compare sequential propagation to an absolute-time Radau integration, compare
batched/scalar outcomes, and check likelihood updates and state preservation.
GPU tests compare RK4 to CPU solutions at prior bounds, halve the step, check
full-horizon and conditional gradients, and cross-check sPCE against scalar
CPU likelihood calculations. A seven-method smoke checks ordered actions,
finite scores, shared evaluator truths, strict checkpoint loading and MoE traces.
IEEE9 regression tests run after the active jobs finish.

tools/validate_ieee14_eig.py runs queued GPU validation, all-method smoke, a
bounded particle/contrast sensitivity diagnostic and an estimate-only benchmark.
It writes readiness.json with pass/fail status. A pending or failed receipt is
not a completed validation. CPU tests alone do not certify the CUDA backend.

The sensitivity diagnostic is a fixed-policy numerical check, not evidence that
128 particles/contrasts are adequate for publication or that MoE wins. A separate
multi-seed performance study must check estimator and planner sensitivity.

## Resources

The timing result is measured on LabPC RTX4090. It is not an A100 runtime.
For FASTER, charged SU = wall hours * (effective CPU cores + GPU rate);
effective cores include memory-based charging. The published A100 rate is
128 SU/GPU-hour, T4 is 64, plus CPU charges. Use maxconfig on the actual Slurm
request and a short authorized FASTER timing run before predicting its total.
Source: https://hprc.tamu.edu/kb/User-Guides/AMS/ (checked 2026-09-17).
No HPRC jobs are submitted by readiness validation.

## LabPC validation update — 2026-09-18

The previous GPU convergence failure came from insufficient precision of the
CPU comparison solution. The test reference now uses DOP853 rtol=1e-12 and
atol=1e-14. The convergence assertion was strengthened to require at least an
eightfold error reduction (plus a 2e-10 reference floor) when halving the RK4
step. Measured reductions across prior endpoints/center, two injection buses
and three carried-state stages were about 15–16 fold. Production dynamics,
priors and the 1/640-second step were not changed.

CPU/GPU regression, independent solver comparisons, T=3/4/5 policy gradients,
conditional refinement gradients and the seven-method smoke passed. Receipts:
`experiments/09182026/09182026_ieee14_eig_validated_T345/validation/`.
The particle/contrast diagnostic is not publication-budget certification.

## Default for new runs (2026-09-18)

The continuous EIG CLI now defaults to 1,024 contrasts (ceiling log(1025) = 6.93245 nats). Planner particles remain 128 by default. The validation receipts and campaign commands above document historical 128-contrast runs and retain their original settings. Existing running and queued jobs are unchanged.
