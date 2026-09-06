# Auditing adaptive and non-myopic room before DAD training

Use `tools/audit_mocu_space.py` for the finite-loss 95% quantile protocol.
The historical duration-search audits do not establish whether this protocol
has useful room for DAD, RL-sBOED or Step-DAD; see MOCU_AUDIT_VALIDITY.md.

## What is measured

The audit uses the production action catalog, observation noise, 384-particle
posterior support, separate control bank and terminal quantile implementation.
Its fast scoring metric is **candidate-grid bank regret**:
`u + 20 * max(U_grid - u, 0) - U_grid`.
This is a model-space diagnostic. It is not the continuous-oracle realized
operational regret reported in final method evaluation. Grid discretization
can affect rankings when under-control is multiplied by 20. A positive screen
requires continuous-oracle validation before a performance claim.

Three paired comparisons answer different questions:

- Fixed minus Myopic: adaptation achieved by the one-step planner.
- Myopic minus lookahead: additional benefit achieved by planning ahead.
- Fixed minus lookahead: their combined benefit.

Positive differences favor the adaptive/planning policy. A changing terminal
control, many different selected sequences, or improved inference alone does
not establish any of these gains. A 95% posterior quantile is not a guarantee
of 95% physical coverage under model error or finite posterior support.

## Efficient sequence

1. Run a cheap screen before any neural training. The tool batches candidates
   and posterior fantasies on GPU. It calibrates Fixed with multistart greedy
   search on independent in-support simulations, bypassing the expensive
   production CPU calibration. The context's temporary trivial-Fixed warning
   refers only to that bypass; the evaluated Fixed policy is calibrated here.
2. Inspect paired gain intervals, control mass, response redundancy, posterior
   effective sample size (ESS), and posterior versus validation loss. Low ESS
   warns about finite-support collapse. Highly correlated actions suggest
   redundancy but do not prove absence of decision-relevant complementarity.
3. Check planning convergence on screening systems: increase inner/outer
   fantasies and first candidates, and separately strengthen Fixed calibration.
   Receding two-step planning only looks two observations ahead at each step;
   it cannot rule out a deeper T-step opportunity. Candidate pruning can miss
   complementary probes. `--first-candidates` equal to the action count removes
   that pruning. There are 54 IEEE9 and 84 IEEE14 actions in the current banks.
4. Freeze the scientific configuration and planning budgets before inspecting
   the second validation half. Confirmation uses that separate half. Once
   inspected, it is consumed: subsequent tuning needs fresh validation for a
   confirmatory claim. The current development reruns are exploratory.
5. Only advance promising, stable settings to continuous-oracle validation
   and the prescribed trained-method comparison. Final test observations are
   never used by this audit's policies or scoring. Existing context loading
   may still validate the test bank's integrity.

```bash
python tools/test_mocu_space_audit.py
python tools/audit_mocu_space.py --config configs/ieee9_mocu.yaml --horizons 2,3,4,5 --phase screen
python tools/audit_mocu_space.py --config configs/ieee9_mocu.yaml --horizons 2,3,4,5 --phase confirm
# Planning-convergence diagnostic on the SCREEN half, not a search for winners:
python tools/audit_mocu_space.py --config configs/ieee9_mocu.yaml --horizons 3 --phase screen --inner 64 --outer 32 --first-candidates 54 --calibration 512 --fixed-restarts 8
```

Replace the config and use 84 candidates for IEEE14. Use the `mocu_optimized`
environment on labpc. Screen defaults are inner=8, outer=4, first=6;
confirmation defaults are 32, 16, 12. Both use all feasible second actions.
The second action is selected on one fantasy set and its value estimated on
another, avoiding the optimistic minimum of noisy second-action estimates.
Policy evaluation uses separate observation noise, with shared action/step
noise across methods for paired comparisons. No repeated actions are allowed.

Each run records resolved config/source hashes, support/validation hashes,
budgets, seeds, validation indices, frozen Fixed sequences, raw per-system
controls/losses/sequences, JSON summaries and a Markdown diagnostic report.
Bootstrap intervals average the three noise seeds within each validation
system and resample systems; they are not based on 192 independent systems.
They are unadjusted per comparison and conditional on the banks, planning
seeds and Fixed calibration. They do not include training uncertainty.

## How to interpret a weak result

`material_gain_not_shown_by_this_planner` concerns this approximate planner,
not an impossibility theorem. `inconclusive` calls for a better-resolved audit,
not automatic rejection of the scientific problem. The default practical
reference, max(0.0001, 5% of prior Bayes risk), is a diagnostic convention;
set a domain-relevant minimum effect before confirmatory experiments.
A positive lookahead-versus-Fixed result can shrink with a stronger Fixed
baseline. A positive planner result also does not guarantee a neural policy
can learn the strategy.

If a converged audit remains weak, examine physically motivated changes to
which machines/locations are probed, observation timing, and uncertain control-
relevant parameter combinations. Preserve the original setting as a neutral
baseline and document every candidate, including failures. Changing only the
number of durations may add redundant actions without useful branches.
Noise changes require a measurement-based justification; do not increase
noise or select cases merely to make DAD win. Hold the chosen 95% quantile
loss fixed while studying these experimental-design changes.

## September 6 development results

The corrected independent-second-stage audit was run on IEEE9 and IEEE14,
T=2,3,4,5, using 64 off-support validation systems and three noise seeds.
These are exploratory validation diagnostics, not final method tables and
not an untouched confirmatory study. The same validation halves were reused
while developing the audit implementation. No neural training was launched.

IEEE9 planning/calibration/evaluation across all four horizons took about
93 seconds on the labpc RTX 4090, excluding context loading. The T=2
Fixed-minus-Myopic gain was 0.00344 with an unadjusted paired 95% interval
[0.00031, 0.00708]. This is a tentative adaptation signal against approximate
Fixed. Its Myopic-minus-lookahead interval crossed zero. At T=3,4,5, all
non-myopic intervals also crossed zero; T=5's mean non-myopic gain was only
0.00094 with interval [-0.00386, 0.00568]. These data do not establish a
reliable additional lookahead advantage or a general absence of DAD room.

Source artifact on labpc:
`experiments/09062026_164546_ieee9_mocu_space_audit_confirm_Uctrl_T5_Nobs5_sigma0p005/`.
Earlier `finite_loss_adaptive_space_v1` artifacts used the minimum of nested
noisy second-stage estimates. Keep them as development records; use the v2
independent-second-stage estimator for new work. Independent external policy
evaluation was present in both versions. A v1 field named `adaptive_gain`
means Fixed-minus-lookahead, a combined effect; v2 additionally reports the
separate `myopic_adaptive_gain`.

IEEE14's corresponding four-horizon audit took about 143 seconds, excluding
context loading. Every non-myopic interval crossed zero. At T=5, the gain
was -0.00057, with interval [-0.00302, 0.00104]. Its positive Myopic-versus-
Fixed means also all had intervals crossing zero. Source artifact:
`experiments/09062026_164721_ieee14_mocu_space_audit_confirm_Uctrl_T5_Nobs5_sigma0p005/`.

Additional T=3 screens with structural/ESS diagnostics found action-response
energy participation ranks of 1.86 for IEEE9 (54 actions) and 1.23 for IEEE14
(84 actions). About 22% of IEEE14 action pairs had theta-centered response
cosine above 0.995, compared with about 1% in IEEE9. This suggests investigating
physically distinct probes before merely adding durations. These ranks are
computed from full observation responses, not a proof about MOCU value.
Median terminal ESS for lookahead was about 162/384 and 96/384 respectively;
these particular screens do not indicate posterior-support collapse.
The screens took 0.84 and 1.25 seconds after context initialization.

Their artifacts are:
`experiments/09062026_165017_ieee9_mocu_space_audit_screen_Uctrl_T3_Nobs5_sigma0p005/`
and
`experiments/09062026_165019_ieee14_mocu_space_audit_screen_Uctrl_T3_Nobs5_sigma0p005/`.
Six regression tests passed, including CPU/CUDA planning agreement,
uninformative observations, complementary XOR probes, batching invariance,
response redundancy and repeated-seed bootstrap clustering.

The present evidence supports caution about assuming a DAD-family advantage.
It does not establish that the problem has no adaptive/non-myopic space.
The next scientific check is planning/Fixed budget convergence on screening
systems, followed by fresh validation if experimental settings are changed.

## Corrected IEEE9 duration-combination search

`tools/audit_ieee9_duration_combos.py` reuses all 281 durations in
`data/ieee9_duration_dense_0p01`, verifies exact latent-row identity and matching
production observation curves, and gets required control from the production
control extension rather than the historical dense bank's U array.
The default search screens 32 reproducibly chosen six-duration sets, including
the current baseline and an evenly spaced set, at T=3. It freezes the three
best balanced combined/non-myopic candidates, retaining the baseline as well,
then evaluates them at T=2,3,4,5 on the other validation half with larger
planning budgets. Each phase uses seeds 101,202,303 and all 384 fit particles.
This is a sampled exploratory search of C(281,6), not exhaustive optimization.

Every screened candidate and every confirmation result is saved, alongside
raw paired arrays and dense-observation/control/latent hashes. Confirmation
reports both ordinary diagnostic intervals and Bonferroni-adjusted bootstrap
intervals across all candidate/horizon/contrast comparisons. Selection on
screening means can overfit; separate validation mitigates that, but the
validation bank has already been used in earlier development. It therefore
still cannot support a fresh confirmatory or publication claim.

On Grace submit `sbatch hprc/ieee9_corrected_duration_audit.slurm`; it requests
one A100, 4 CPUs, 32 GB and a 2-hour time limit. It first runs the audit
regressions and objective-alignment check. No DAD training is launched and
no production duration catalog is changed by this search.
