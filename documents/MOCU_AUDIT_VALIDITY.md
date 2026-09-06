# Validity of historical MOCU audits after the finite-loss correction

The production IEEE9/IEEE14 protocol now uses alpha=0.05, zero margin,
under-control penalty 20, zero event penalty, and the quantile decision.
Historical audit outputs must be judged against the decision rule they actually
used. They have not been rerun or rewritten.

## Latest IEEE9 duration audit

Artifact: `experiments/09042026_210922_ieee9_mocu_duration_fullgrid_audit_Uctrl_T2_Nobs5_sigma0p005/diagnostics/summary.json`.

It explicitly identifies its objective as `Yoon IBR MOCU identical to
training/evaluation`. The current historical search implementation in
`tools/bank_sweeps/search_ieee9_mocu_duration_sets_full.py` installs a terminal
function using `ibr_max_u_ctrl` and `belief_mocu`; alpha, margin and finite-loss
penalties are intentionally ignored by that function.

Its zero passing finalists do not validate or invalidate the corrected
finite-loss protocol. The report remains a historical observation about the
old rule and its screening gates. Its search considered all 281 durations as
eligible but sampled combinations: 20,281 global candidates, 120 global exact
evaluations, 120 local exact evaluations, and 20 strict finalists. It did not
exhaust the 647,992,090,956 six-duration combinations.

The audit used a 96-particle support and horizon two, whereas the inspected
production experiments use a 384-particle posterior support after reserving
validation systems and horizons three or greater. Even with matching losses,
these are exploratory screening results, not production method comparisons.
The positive-advantage and branch-value thresholds are research-selection
criteria, not definitions of computational correctness or physical safety.

## IEEE14 and future audit invocations

The historical IEEE14 script uses a finite under-control loss and a quantile,
but its default alpha is 0.01 while its penalty default is 20. That pair is not
the aligned 95% quantile protocol. Both duration scripts also have a default
control grid different from the current production grid, and they read U from
the dense physical bank rather than automatically checking the current separate
control-extension identity. Saved invocation details, physics, U bank, grid,
noise and support must be verified before interpreting any particular run.
Some historical metadata omit alpha/grid, so the script defaults alone cannot
establish which CLI overrides were used in a historical run.

These historical search scripts have NOT been migrated by the production MOCU
fix. Do not launch them expecting automatic agreement with the corrected YAML.
A new audit must explicitly share the production terminal loss/decision and
record its full resolved protocol, source identity and bank identity. Candidate
selection and final validation should use separate random draws; final claims
need the prescribed training/evaluation seeds and independent physical safety
assessment. Selecting cases because adaptive methods win is not itself an
unbiased evaluation of those methods.

## What the new checks establish

The eight regression tests, CPU/Torch agreement, bank-quality checks, and
integration smoke establish correctness of the tested code paths. The new
64-rollout validation preflights establish that the corrected terminal
decisions vary on the existing IEEE9/IEEE14 banks. They do not establish
publication performance, 95% held-out physical safety, or an adaptive method's
advantage over Fixed/Myopic. Existing maximum-rule trained models and result
tables are not converted into finite-loss results by changing their labels.

## Replacement adaptive/non-myopic audit

Use `tools/audit_mocu_space.py` and see [MOCU_SPACE_AUDIT.md](MOCU_SPACE_AUDIT.md)
for GPU batching, separate adaptation/planning comparisons, paired validation,
protocol limitations and the September 6 development results. Its fast metric
is candidate-grid bank regret; final continuous-oracle evaluation remains
a separate requirement.
