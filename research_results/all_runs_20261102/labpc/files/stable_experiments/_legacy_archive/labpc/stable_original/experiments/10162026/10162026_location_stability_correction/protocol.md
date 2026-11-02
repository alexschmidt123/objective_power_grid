# Controlled location-finding stability correction

The original five-seed campaign is preserved. All corrections are in an isolated source tree; active SIR uses unchanged original source.

## Order

1. Focused correctness tests and CPU root-launcher smoke tests.
2. Existing-checkpoint hard/relaxed diagnostics (512 paired development cases, unchanged L1M evaluator), corrected ScoreBED validation replay.
3. If the diagnostic gate supports instability beyond hard discretization, queue four fresh full-budget development runs: MoE102, SwitchMoE102, MoE103, SwitchMoE103.
4. Each model retains1M episodes, L1024, existing architecture/optimizer/clipping/regularization and 30%/95% schedule. Only terminal gamma changes0.001 to0.01.
5. Evaluate final checkpoints on2000/L1M/1001 paired cases. Compare raw router gradients, hard/relaxed gaps, utilization and gate behavior. No MSE selection.
6. Expand to remaining101/104/105 only if every predeclared criterion in protocol.json passes; otherwise retain negatives and stop.

## Decisions fixed before candidate training

For each model: seed103 EIG gain >=0.5 nats with paired trajectory MC lower95% >0, and final20% router p99 <=10% of original. Seed102 gain >=-0.1 with paired MC lower95% >-0.2. The absolute hard/relaxed mean gap may increase by at most0.2 on the512-case diagnostic subset. These are explicit development go/no-go tolerances, not universal journal significance thresholds. Across-training-seed claims require complete declared replication; subset MC intervals are not training uncertainty.

## Resource handling

No new GPU training overlaps the running original SIR campaign. The queue waits for that coordinator and other GPU jobs; it checks free disk and source hashes before each root run. It does not cancel current work. No automatic training retries or architecture search. Exact failed-attempt costs remain recorded.

## ScoreBED

Bounded metadata is propagated into intermediate candidate exports. Native action-reference and complete validation/deployed rollout parity tests are required. Existing invalid curves remain preserved and flagged; only available checkpoint weights can be rescored. A metadata-only repair does not justify full retraining.

## Reproducibility

See protocol.json, provenance.json, original_source_manifest.json, changes.json, *.patch, corrected_source_manifest.json, diagnostics/, correctness_tests.log, smoke/, queue_status.json, candidate_registry.json and reports/. The original study and reserved final evaluation seed remain untouched.
