# Project instructions

## Repository layout and manuscript editing

- Keep exactly these files at the root of `documents/`:
  `objective_driven_boed_manuscript_framework.tex`, `sBOED_design.tex`,
  `objective_driven_boed_refs.bib`, and the existing
  `objective_driven_boed_manuscript_framework.pdf`. Keep `images/` and `papers/`.
  Do not add audit reports, temporary notes, LaTeX fragments, or build files there.
- Edit LaTeX and BibTeX only by default. **Never compile or rebuild the manuscript
  PDF unless the user explicitly asks.** Preserve its existing bytes during
  cleanup and synchronization. For ordinary edits, check citation keys, labels,
  inputs, and matching equations using source-only validation.
- Both standalone LaTeX documents contain a marked finite-loss MOCU block.
  Update both copies together; `tools/check_repository.py` checks their identity.
- Keep cited primary-source PDFs in `documents/papers/`; verify PDF content and
  title, and record source URL, filename, and SHA-256 in that folder's
  `manifest.json`. BibTeX `file` fields are relative to `documents/`.
- Figures belong in `documents/images/`. Preserve the IEEE9/IEEE14 style when
  adding networks: numbered circles, black branches, yellow slack, green
  machine terminals, blue algebraic buses, clear legend and source. A PV bus
  does not automatically imply a governed dynamic generator.
- Only five YAML files may exist in `configs/`: `ieee9_eig.yaml`,
  `ieee9_mocu.yaml`, `ieee14_mocu.yaml`, `ieee30_mocu.yaml`, and
  `sir_ode_eig.yaml`. No subfolders, host variants, temporary sweeps, or backups.
  Store any necessary resolved experimental variation inside its stamped
  `experiments/` directory, derived from and linked to a canonical configuration.
  Do not change physics just to accommodate a host's paths.
- `tools/` is for reusable bank helpers, diagnostics, audit engines, reference
  data, and regression checks. Shared audit functions belong in `tools/audits/`,
  tests in `tools/tests/`. Remove obsolete one-off rerun/search scripts after
  consolidating useful functions. Do not duplicate the production objective
  inside a diagnostic or monkey-patch its terminal rule.
- `hprc/` contains only reusable cluster environment and Slurm wrappers.
  Job IDs, seeds specific to one campaign, and completed-run paths belong in
  run manifests, not permanent wrappers. Never edit running source snapshots.
- Keep this `AGENTS.md` as the authoritative reusable workflow guide. Keep
  `README.md` brief: introduction, installation, and basic run instructions.
  Historical audit reports remain in their experiment artifacts or Git history.

## Mandatory formal-run entrypoints

- **Every formal data-generation, training, evaluation, or publication run must
  start through `run.sh`, `sweep_run.sh`, or a maintained `scripts/*.sh` entrypoint.**
  This rule applies on labpc, Grace, other hosts, and inside Slurm jobs.
- Do not launch a formal run by invoking `python -m src.experiment`, a training
  function, or a temporary Python/shell launcher directly. The maintained shell
  entrypoints may call Python internally. If a reusable operation is missing,
  implement and verify the appropriate `scripts/*.sh` entrypoint first.
- Slurm wrappers may select resources, activate the environment, enter a frozen
  source directory, and forward arguments to the maintained shell entrypoints.
  Do not implement another training/evaluation pipeline in `hprc/`.
- Unit tests and bounded diagnostic audits are not formal method comparisons.
  Use `scripts/check.sh` and `scripts/audit.sh` for their reusable workflows;
  label smoke/audit outputs so they cannot be mistaken for publication results.
- Do not submit new long-running experiments merely to verify a cleanup or
  documentation change. Do not terminate or modify existing jobs without task
  authorization. A new run uses a new output directory and records its command.

## Scientific configuration and validity

- Main application order: IEEE9 MOCU, then IEEE14 MOCU, then IEEE30 MOCU.
  SIR ODE EIG is an implementation benchmark. Do not expand IEEE14/IEEE30 bank
  work while the user's current priority remains IEEE9 unless requested.
- Primary MOCU uses `robust_rule: quantile`, `alpha: 0.05`, zero safety margin,
  under-control penalty 20, zero binary event penalty, and aligned grid support.
  Total loss is `L(u,U)=u+20*max(U-u,0)` and regret is `L(u,U)-U`.
  The 95% quantile is a Bayes action for this loss; it is not a worst-case
  maximum or a guarantee of 95% out-of-sample physical safety.
- Training, Fixed calibration, Myopic fantasies, Step-DAD refinement, and
  posterior evaluation must use the identical loss and decision rule.
  Snapped quantiles are aligned only when stored requirements already lie on
  the candidate grid. Otherwise use continuous controls or explicitly minimize
  the finite loss over the available grid. Preserve the alignment guard.
- Primary `mean_mocu` in schema `realized_operational_regret_v2` means held-out
  realized regret against a refined continuous-control oracle.
  `mean_posterior_mocu` is a separate bank-posterior diagnostic. Never combine
  older posterior-only or hard-maximum results under a shared metric label.
  A continuous-oracle comparison also includes control discretization error.
- Report empirical physical safety, nadir/ROCOF violations, control magnitude,
  posterior ESS, and regret separately. Keep failed/ineligible method outcomes;
  distinguish a safety point estimate from its statistical confidence interval.
- The scalar requirement model assumes a feasible control range and monotone
  safety. Check both. Do not discard or resample infeasible physical systems
  to improve safety scores. A single-contingency run is not N--1 validation.
- IEEE30 is the Demetriou et al. (2017), DOI 10.1109/JSYST.2015.2444893,
  modified dynamic reference: 36 electrical buses, six GENROU machines,
  IEEET1 excitation, and BPA_GG governors only on the two generators.
  Original buses 1,2,5,8,11,13 map to terminals 31--36; the last four machines
  are synchronous condensers. The proposed latent vector is six H and two R
  coordinates. Prior widths are study assumptions, not measured data.
  Machine-base conversion is `M_i=2*H_i*(S_i/S_base)/(2*pi*f_base)`.
  Do not assign inertia or governor response to every bus, or replace a
  condenser's excitation gain by active-power droop. The full dynamic backend
  and AC initialization are pending; retain the explicit unsupported-model
  guard and do not launch this YAML with the reduced swing solver.

## Data generation and reuse

1. Activate the installed environment and inspect the canonical YAML, physical
   model, observation definition, action catalog, control scenario, units,
   and bank paths. Use the finite-loss alignment and repository checks first.
2. Generate or validate data through, for example:
   `bash scripts/data_generation.sh --config configs/ieee9_mocu.yaml --experiment_type objective_based --T 3 --N_obs 5 --noise_sigma 0.005 --seed 101`.
   Do not invoke the internal bank helpers directly for a formal generation.
3. The reusable pipeline ensures the configured master bank, extracts the
   active subset without resimulation, then validates/builds the separate
   control extension. `data.reuse_bank_dir` identifies the master bank.
   For IEEE9 the full grid has 281 durations, 0.20--3.00 s in 0.01 s steps,
   crossed with nine buses: 2,529 actions. Six selected durations give 54
   actions; this is not a selection of six arbitrary bus--duration pairs.
4. A dense grid is a master bank, not an exact continuous-duration simulator.
   Every on-grid duration combination must be extractable as a subset.
   Check complete duration-by-bus coverage and exact observation-column order.
   IEEE14 needs 281*14=3,934 actions for its analogous master bank.
5. Validate theta values and row ordering across observation/control banks,
   configuration/source hashes, shapes, finite trajectories, grid, and units.
   File existence alone is insufficient. Changing the decision rule alone
   need not regenerate physical responses; changed physics invalidates reuse.
6. Preserve transactional bank writing: stable sibling lock, private staging,
   validation/completion manifest, publication only after success, and retained
   replaced banks for existing memory maps. Do not delete failed staging
   evidence or old banks while readers are active.
7. Keep training/validation/final-test roles explicit. Fit support and neural
   training may share their declared training bank; Fixed calibration uses
   training simulations. Final test outcomes must not guide selection.
   Preserve validation preflight and inspect constant terminal decisions
   before expensive training; do not disable it merely to obtain a run.

## Experiments, reruns, and Grace

- A single matched pilot:
  `bash run.sh --config configs/ieee9_mocu.yaml --experiment_type objective_based --method dad,rl_sboed,step_dad,myopic,fixed,random --T 3 --N_obs 5 --noise_sigma 0.005 --seed 101 --eval-seeds 1001`.
- A full main-study sweep:
  `bash sweep_run.sh --configs ieee9_mocu --experiment_type objective_based --T 3,4,5 --N_obs 5 --noise_sigma 0.005 --seed 101,202,303 --eval-seeds 1001,1002,1003,1004,1005`.
- Separate training uses `scripts/training.sh`. Rerun evaluation through
  `scripts/evaluation.sh --config ... --experiment_type ... --exp-dir ... --T ... --N_obs ... --noise_sigma ... --seed ... --eval-seed ...`.
  Verify the full configuration and checkpoint identity before reuse;
  Step-DAD reuses its matching DAD checkpoint and adds online refinement.
  Preserve original results and record replacement provenance for reruns.
- On labpc the usual environment is `mocu_optimized`; on Grace the reusable
  environment wrapper loads GCCcore/12.2.0, Python/3.10.8, CUDA/12.1.1 and
  `/scratch/user/g.lin/venvs/objective_power_grid` (override with `BOED_VENV`).
  Never run GPU simulation/training on login nodes.
- Freeze source/configuration before submission and record commit, content
  hashes, resolved arguments, and Slurm IDs. Include `src`, `tools`, `configs`,
  `scripts`, `hprc`, `documents`, `run.sh`, `sweep_run.sh`, `AGENTS.md`, and
  `README.md` when preparing an audit archive used by repository checks.
  Symlink shared data into the snapshot; output belongs outside the snapshot.
- Submit formal work with
  `sbatch --output=/absolute/run/logs/%x-%j.out hprc/experiment.slurm /absolute/source_snapshot run [run.sh arguments]`.
  Create the log directory before submission. Entry choices are `run`, `sweep`,
  `generate`, `train`, `evaluate`, and `visualize`. Override resource requests
  at submission time; do not make a new wrapper for each seed or horizon.
- Report queue delay separately from estimated compute time. Base ETA on
  measured stage throughput when available; a Slurm wall limit is not an ETA.
  Inspect `squeue`, `sacct`, logs, and completion manifests before claiming
  completion. A zero exit code without expected method/seed outputs is insufficient.

## Reusable audits

- `bash scripts/audit.sh alignment` checks production objective agreement;
  `... preflight --config configs/ieee9_mocu.yaml` checks decision sensitivity.
- `bash scripts/audit.sh space --config configs/ieee9_mocu.yaml --horizons 2,3,4,5 --phase screen`
  runs a bounded planner screen. Use `--phase confirm` only with declared
  unconsumed validation. Increase fantasies, first-action coverage, and Fixed
  calibration to investigate numerical convergence on screening systems.
- `bash scripts/submit_audit.sh --archive /absolute/source.tar --commit COMMIT --runtime /absolute/runtime`
  submits the reusable IEEE9 master search on Grace. `--previous` optionally
  reuses candidates from a specified previous campaign; no historical run path
  is hard-coded. The stages are prepare, fresh, confirm, finalize, with Slurm
  `afterok` dependencies and confirmation concurrency controlled by an argument.
- Master search samples global catalogs covering all 281 durations, refines
  local neighbors, includes the unchanged baseline, and selects by both low
  loss and estimated adaptive/planning room. It is planner-based selection.
  Label the result best found, not global optimum over C(281,6).
- Freeze finalists before fresh physical validation. Reproduce known probe
  curves and compare batched/scalar continuous oracles (tolerance 1e-4).
  Use the union of finalist actions for fresh validation; do not regenerate
  the full master for each subset. Do not reuse consumed validation for new
  confirmatory selection without new independent data.
- Distinguish Fixed-minus-Myopic adaptation, Myopic-minus-lookahead planning,
  and Fixed-minus-lookahead combined gain. Branch counts alone prove none.
  Approximate Fixed and rolling two-step lookahead are not globally optimal;
  a null result is not a theorem that DAD has no useful full-horizon strategy.
- Select second actions and estimate their values with independent fantasy
  sets. Pair true systems and action/step noise across methods. Average noise
  replicas within theta before theta-cluster bootstrap; adjust confirmatory
  intervals for all prespecified finalists/horizons/contrasts. Disclose that
  intervals conditional on a bank do not measure bank-generation uncertainty.
- Keep all candidates, failed checks, raw arrays, frozen choices, budgets,
  seeds, oracle brackets, configuration, source hashes, and reports inside
  the campaign directory. Never change priors or durations after inspecting
  final test results merely to make a DAD-family method win.

## Results and visualization

- All generated run outputs live in stamped `experiments/` directories:
  resolved configuration, source/command provenance, local checkpoints,
  training logs, diagnostics, evaluation seed subdirectories, raw per-system
  records, summaries, and completion manifests. Keep banks under `data/`.
  Do not scatter new logs, results, configs, or caches into repository roots.
- Verify every requested method, horizon, training seed and evaluation seed;
  finite scores; metric schema; shared physical systems/noise; and runtime
  units. Never fabricate missing cells or silently drop failed seeds.
- Replot existing sweeps using `bash sweep_run.sh --configs ... --T ... --N_obs ... --noise_sigma ... --seed ... --experiment_type ... --plots-only`.
  For explicit historical run directories, use
  `bash scripts/visualization.sh results --exp-dir /absolute/run1 --exp-dir /absolute/run2 --T 3,4,5 --N_obs 5 --noise_sigma 0.005 --experiment-type objective_based`.
  Old SIR folders retain the `sir_ode` stem: pass explicit directories instead
  of renaming their records to the new `sir_ode_eig` config stem.
- Save plot provenance and the underlying tables. Do not pool incompatible
  duration catalogs, objectives, noise settings, or model versions.
- Render the IEEE30 reference topology with
  `bash scripts/visualization.sh network --config configs/ieee30_mocu.yaml --output documents/images/ieee30_diagram.png`.
  This draws a source-checked network; it does not validate or execute the
  pending full dynamic simulator. Do not generate technical connectivity with
  an image model or infer machines from all bus labels.

## Performance-result tables

These rules are mandatory for every final performance table:

- Create one table per metric. EIG, MOCU, offline time, and online time must
  never be combined into one table.
- Use methods as rows.
- Use experiment horizon `T` as columns.
- Format every numeric cell as `mean ± std`, using sample standard deviation.
- Final publication tables merge all training and evaluation seeds into one
  table; do not publish a separate table for each training seed.
- For trained methods (DAD, RL-sBOED, and Step-DAD), compute EIG, MOCU, and
  online-time mean and sample standard deviation over all 15 crossed results:
  3 training seeds (`101`, `202`, `303`) × 5 evaluation seeds (`1001` through
  `1005`).
- For non-training methods (Myopic, Fixed, and Random), training-seed copies
  are duplicate evaluations. Deduplicate by evaluation seed and compute mean
  and sample standard deviation over the 5 unique evaluation seeds. When a
  repeated runtime measurement differs, average its repeated copies within
  each evaluation seed before computing the five-seed summary.
- Offline time is not evaluation-seed dependent. For trained methods, compute
  mean and sample standard deviation over the 3 training seeds; do not repeat
  an offline time five times. Apply the same run-level treatment to any
  non-training calibration time that is measured once per experiment run.
- Treat the 15 trained-method combinations as descriptive crossed results, not
  as 15 independent training runs. State the `3 training × 5 evaluation`
  design in the table caption or accompanying text.
- Bold the best method value within each comparison column (`T`). For EIG,
  higher is better; for MOCU and both time metrics, lower is better. Bold all
  tied winners.
- Missing training or evaluation seeds make a cell incomplete; show an em dash
  rather than presenting it as a final result.

For an ablation table, the columns may represent the ablated variable instead
of `T`. The one-metric-per-table and `mean ± std` rules still apply.


## Configurable posterior coverage for MOCU

- Set `control.posterior_coverage: q` with `0 < q < 1` in the experiment YAML.
  This is a posterior coverage preference, not a physical safety standard.
  Express the loss as `L(u,U)=u+max(U-u,0)/(1-q)` and the decision as the
  posterior q-quantile. Do not introduce separate named tail or penalty
  parameters in the manuscript or user configuration. The loader supplies
  legacy internal loss fields from `q`; it sets no empirical safety threshold. Derived legacy keys
  are overridden when this parameter is present; old snapshots without it
  keep their original behavior. Quantile control, zero margin, and zero
  binary violation penalty are required. Physical frequency and RoCoF limits
  are separate and are not changed by this parameter.
- Retrain each coverage setting through `sweep_run.sh` in an isolated source
  snapshot; reuse matching physical banks, not checkpoints or Fixed caches
  from a different loss. Record the input ratio and resolved loss parameters.
- Compare methods within each ratio; raw MOCU across ratios has a different
  loss scale. When the user asks for MOCU, report `mean_posterior_mocu` only
  by default; do not relabel `mean_mocu` (held-out regret) as posterior MOCU.


## Explicit approval before task submission

- User instruction: never submit any task without explicit user approval. Before
  submitting cluster jobs or launching experiment/audit tasks, present the exact
  settings, methods, seeds, job count, estimated duration, and SU cost for review
  and wait for approval. A parameter discussion or request to make a parameter
  configurable is not submission approval. Read-only status checks, preparation,
  and requested cancellations can proceed. Do not expand an approved run scope.


## Explicit approval before running or editing code

- Never run any code or edit any code without the user's explicit approval.
  This includes scripts, shell commands, tests, diagnostics, experiments, and
  code changes. Describe the proposed action and wait for approval before
  executing it. Do not infer approval from a discussion, suggestion, status
  question, or earlier approval for a different action. Approval applies only
  to the action and scope explicitly authorized by the user.
- This rule takes precedence over any earlier instruction allowing autonomous
  execution, read-only command checks, preparation, tests, or code edits.


## Joint MOCU and safety reporting

- Report terminal posterior MOCU and empirical physical safety rate as separate
  metrics for each method; posterior MOCU is primary and safety is diagnostic. Posterior coverage is an input preference;
  empirical safety is an output, not an engineering acceptance threshold.
- Safety requires both frequency-nadir and RoCoF limits for the declared
  monitoring interval and contingency set. Average repeated outcomes within
  each physical system before averaging systems; preserve safe/unsafe counts,
  number of outcomes, and number of distinct physical systems. Confidence
  intervals must respect physical-system clusters and stated sampling assumptions.
- The human-readable MOCU column uses mean_posterior_mocu, never the legacy
  mean_mocu field (held-out regret). Keep every method visible, including poor
  safety outcomes; legacy validity flags do not establish engineering approval.
- Never edit existing submitted source snapshots or historical result files
  to apply reporting changes. New code applies to future authorized runs.


## Posterior-MOCU selection protocol

- Primary training rewards, validation checkpoint selection, result ranking,
  plots, and publication tables use terminal posterior MOCU. Safety rate and
  control magnitude are supporting diagnostics, with no automatic safety gate.
- Coverage does not set an empirical safety threshold. Deprecated safety-gate
  keys do not affect checkpoint selection. Optional MoE diversity ablations
  must remain explicitly labeled; standard methods use zero diversity weight.
- The legacy evaluation mean_mocu column still denotes realized regret for
  schema compatibility. Never use it as a fallback for missing posterior MOCU.
- Historical submitted runs used the earlier regret-based checkpoint selection
  and safety gate. Their frozen protocols must be disclosed, not retroactively
  described as using the corrected posterior-only checkpoint rule.


## Independent MSC objective

- `--experiment_type msc_based` selects posterior minimum safe control, alongside
  existing `objective_based` (MOCU) and `eig_based` (EIG). This adds an objective;
  do not rename historical MOCU runs or overwrite their results/checkpoints.
- MSC is restricted to IEEE9, IEEE14 and IEEE30 configurations. IEEE30 remains
  blocked by the existing unsupported dynamic-backend guard; selecting MSC does
  not authorize or validate the reduced-swing surrogate for that model.
- Reuse the canonical grid YAMLs and their physical observation/control banks.
  `u_optimal.npy` / legacy `psi_star.npy` stores minimum-safe requirements;
  `control_safe.npy` stores safety across the candidate grid. There is no need
  for another physics bank solely because the probe objective changes.
- MSC uses zero margin, the discrete admissible control grid and configured
  `control.posterior_coverage=q`. It minimizes expected terminal selected
  control, E_D[u_MSC(D;q)], subject to the posterior coverage constraint in each
  decision. It has no shortfall penalty in its training/design score.
  A lower MSC is not a physical safety certificate. Quantile expectation is
  not generally monotone under information; do not promise a method advantage.
- Reject nonfinite/infeasible requirements and bank safety tables inconsistent
  with the assumed monotone scalar safety threshold. Never discard unsafe
  systems to make a result look better. Frequency/RoCoF safety is evaluated
  independently on held-out true systems.
- DAD rewards negative terminal MSC; RL-sBOED uses telescoping MSC reduction;
  Fixed, Myopic and Step-DAD optimize the same terminal MSC. Random uses the
  same final controller. Use independent `training.msc_based` settings.
- MSC checkpoints record objective, posterior coverage and terminal-rule hash;
  reject cross-objective or mismatched-rule reuse. Fixed caches include the
  objective in their fingerprint. MSC folders have an MSC token.
- For MSC, primary `mean_msc` is the selected control averaged within physical
  systems and then across systems; `mean_oracle_msc` uses true parameters and
  the refined numerical oracle. Report physical safety alongside MSC, and
  preserve actual `mean_posterior_mocu` as a distinct secondary diagnostic.
  Do not label MSC as monetary cost or energy without a physical conversion.
- Keep one metric per publication table and the existing crossed-seed rules.
  These instructions override MOCU-only reward/ranking instructions for MSC;
  they do not change the MOCU or EIG protocols or their submitted snapshots.
- MSC implementation work is not run authorization. Formal experiments still
  require the user's explicit approval and maintained shell entrypoints.


## Shared bank layout

- Store reusable IEEE9 data under `data/ieee9/probe_master/`,
  `data/ieee9/probe_subsets/<bank-id>/`, and
  `data/ieee9/control_banks/<bank-id>/`. MOCU and MSC share physical control
  bank structures; coverage and objective belong to run configuration.
- Bank IDs preserve physical/data provenance; validate physics, safety limits,
  control grid and identical parameter rows before reuse. A matching folder
  name alone does not establish compatibility.
- Put run configurations, checkpoints and results in stamped experiments.
  Reference shared bank paths; do not generate reusable banks inside a run.
- Preserve old paths as compatibility symlinks when relocating existing banks;
  never rewrite frozen source snapshots or historical bank metadata.
  `data/ieee9/layout_manifest.json` records old and new locations.

## MSC development-space audit

- Use `bash scripts/audit.sh msc-space --config <resolved-MSC-yaml> --output <stamped-experiment-directory>` for MSC diagnostics.
- The audit scores production posterior selected control, never MOCU regret. It compares no probes, Random, calibrated Fixed, Myopic and receding two-step lookahead. Save paired arrays and histories, coverage, budgets and provenance.
- Treat bank-threshold coverage as a proxy, not physical simulation. Existing training-validation systems are development data, not fresh confirmation. Bootstrap intervals cluster repeated noise seeds by physical system.
- A null approximate-planner result does not prove absence of adaptive or non-myopic opportunity. Increasing fantasy budgets and independent confirmation are needed before positive claims.

- To search new MSC duration combinations, use `scripts/audit.sh master` with
  `--objective msc --config <resolved-MSC-yaml>`. The `prepare` stage searches
  the complete master duration catalog; a single `msc-space` audit does not.
  Run `prepare`, `fresh`, each finalist `confirm`, then `finalize` in order.
  Declare horizons, global candidate count and fresh-system count before launch.
  Freeze finalists before simulating fresh validation; retain all candidate
  scores, paired arrays and physics checks. Report MSC and physical safety
  separately. A sampled/global-local search identifies best-found sets, not
  an exhaustive optimum over all six-duration combinations.
