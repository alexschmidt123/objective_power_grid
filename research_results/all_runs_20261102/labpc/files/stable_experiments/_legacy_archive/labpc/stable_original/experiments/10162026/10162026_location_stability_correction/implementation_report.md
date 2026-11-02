# Stability correction: implementation and execution report

Implementation, correctness checks and frozen-checkpoint diagnostics are complete. Full-budget corrected policies have not yet been trained. The resource-serialized coordinator is waiting for the existing SIR campaign/GPU; see live queue_status.json.

## Implemented changes

- Isolated source tree: source/. Original frozen source hash verification passed; historical results and active SIR are unchanged.
- src/research/journal_scorebed_bridge.py: export_candidate preserves location common_design_bounds; intermediate validation uses this export.
- src/research/sparse_moe.py: configurable terminal gamma with exact legacy default0.001.
- src/research/family_training.py: reads terminal gamma and records it in training metadata.
- tools/tests/test_stability_correction.py: metadata/action/native-reference/rollout parity, schedule compatibility, single-factor configuration and acceptance-rule tests.
- tools/stability_diagnostics.py: CPU paired hard/relaxed checkpoint diagnostic and available ScoreBED validation replay.
- tools/stability_campaign.py: guarded, root-launcher-only, serialized, conditional training and paired reporting; no automatic failed-run retry.
- configs/stability/location_gamma001.yaml: terminal gamma0.01; established full-budget configuration otherwise preserved.

## Correctness results

- 64 tests passed;1 historical-binary fixture test skipped because historical binaries are not copied into the isolated source. Actual checkpoint compatibility is additionally verified by reproducing hard actions/EIG for the first16 cases of each of the six real diagnostic checkpoints.
- MoE and SwitchMoE root-launcher CPU smoke tests passed:8 primary episodes each,2 optimizer updates, final-budget checkpoint selection. These are workability tests, not EIG evidence.
- Native ScoreBED saved JAX reference actions match corrected validation; complete Torch validation/deployed rollouts and weights match exactly for seeds102–105.

## Paired diagnostic results

Development subset only: first512 original paired evaluation1001 cases, same L1M contrast bank and common scorer. Not replacement main results.

|Method|Seed|Hard EIG|Relaxed EIG|Relaxed−hard|Paired MC95% interval|
|---|---:|---:|---:|---:|---|
|moe|102|9.75033|9.79363|+0.04330|[-0.00633,+0.09292]|
|moe|103|6.27867|6.29222|+0.01356|[-0.02681,+0.05392]|
|moe|104|10.42646|10.42964|+0.00318|[-0.00524,+0.01160]|
|switchmoe|102|10.11043|10.11983|+0.00940|[-0.03603,+0.05483]|
|switchmoe|103|6.52223|6.57408|+0.05185|[-0.02278,+0.12648]|
|switchmoe|104|10.64331|10.66876|+0.02545|[-0.00389,+0.05478]|

Both failed seed103 models remain poor under relaxed inference. The gate supports a controlled optimization/sharpening experiment, not a claim that gamma alone is the root cause.

## Corrected ScoreBED validation

|Seed|Originally logged unbounded EIG|Correct bounded validation EIG|
|---|---:|---:|
|102|6.21599|7.18030|
|103|6.42771|10.39889|
|104|5.97182|9.72000|
|105|6.01528|9.61980|

These are512-case validation scores using seed771001, not2000-case held-out results. Final held-out ScoreBED results are retained. Available candidate files are final checkpoints; missing historical intermediate checkpoints cannot be reconstructed.

ScoreBED score-training logged tail-loss means are similar across seeds; seed102 is not identified as a score-loss explosion. All seeds show frequent policy-gradient clipping in logged tail updates. This is insufficient to diagnose score-gradient accuracy or justify a selective training rerun. See scorebed_optimization_audit.json.

## Frozen next steps

1. Wait for the original SIR campaign and other GPU jobs; do not interrupt or overlap training.
2. Four fresh1M-episode development runs: MoE102, SwitchMoE102, MoE103, SwitchMoE103. Only terminal gamma0.001→0.01.
3. Evaluate final checkpoints on2000/L1M/1001 with paired case and bank hash verification, gradient diagnostics and512-case relaxed diagnostics.
4. Accept only if all predeclared protocol.json requirements pass. Complete101/104/105 only after acceptance. Negative result stops the candidate without architecture search.
5. No vsOED, ScoreBED or residual training reruns are launched by this campaign. No untouched final seed or fresh-seed confirmation is launched.

## Reproduce / inspect

From the isolated source directory:

```bash
CUDA_VISIBLE_DEVICES= /home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python3 -m pytest -q -rs tools/tests/test_stability_correction.py tools/tests/test_journal_validation.py tools/tests/test_moe_family.py tools/tests/test_journal_campaign.py
CUDA_VISIBLE_DEVICES= /home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python3 -m tools.stability_diagnostics
/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python3 -m tools.stability_campaign
```

The coordinator is already running; do not start a duplicate. Exact per-run commands are in planned_commands.json. Its lock rejects a second coordinator. Queue wait cap7days, minimum free disk10GiB, optimization cap7200s per candidate and root-process wall cap10800s.

No improved candidate EIG is claimed; candidate training is pending.
