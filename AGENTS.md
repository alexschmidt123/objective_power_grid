# Project instructions

## Performance-result tables

These rules are mandatory for every final performance table:

- Create one table per metric. EIG, MOCU, offline time, and online time must
  never be combined into one table.
- Use methods as rows.
- Use experiment horizon `T` as columns.
- Format every numeric cell as `mean ± std`, using sample standard deviation.
- Compute the mean and standard deviation over exactly five evaluation seeds:
  `1001`, `1002`, `1003`, `1004`, and `1005`.
- Preserve the five seed-level values. Do not first collapse them into one
  value, and do not treat systems or rollouts within an evaluation as seeds.
- A cell with fewer or more than five seed values is incomplete and must not
  be included in a final performance table.

For an ablation table, the columns may represent the ablated variable instead
of `T`. The one-metric-per-table and `mean ± std` rules still apply.
