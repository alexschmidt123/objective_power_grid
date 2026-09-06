# Project instructions

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
