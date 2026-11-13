# LabPC validation

End-to-end GPU smoke validation on the actual frozen V2 checkpoint uses reduced posterior/contrast/trial counts and six selection plus six independent test histories at stages0,14,29. Smoke returns are not scientific evidence.

Assertions verify all four experts are removed from every router/fusion consumer: perturbing removed slots, responses and legacy features does not change masked gates or actions; removed gate is exactly zero. Existing validated all-consumer mask is reused without editing frozen code.

Native control continuation matches original V2 at stages0,14,29 within explicit float tolerances. Observed prefixes are exactly preserved, future fields are cleared, observations follow exact likelihood signal plus prescribed common noise, actions obey physical bounds, and single/persistent removals agree on the intervention's first action. A synthetic conditional-importance fixture verifies the selection-only stump can recover known expert switches. Regime is locked before test scoring and checked against final analysis.

The smoke run also checks unchanged actor parameters/buffers and historical/source/checkpoint hashes, no optimizer steps, independent selection/test histories and completed report statistics. Full-run posterior thresholds and analysis are locked in PROTOCOL.md before execution. These checks establish computational correctness, not a positive contribution result.

Posterior refinement scale check, before any EIG outcomes: the initially failed history with 2097152 proposal particles passes unchanged thresholds, ESS281.6265 and maximum weight0.0227493. Zero EIG evaluations/optimizer steps. Receipt: `scratch_runs/hec_v2_expert_contribution_20261113_refined_scale_check.json`. The explicit follow-up uses this fixed larger count for all48 histories and preserves the initial inconclusive run. A separate smoke run verifies the new CLI flag and report/watcher modules parse.
