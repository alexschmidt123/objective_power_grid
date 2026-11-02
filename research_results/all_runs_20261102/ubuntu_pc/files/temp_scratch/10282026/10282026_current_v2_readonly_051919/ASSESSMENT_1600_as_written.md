# Current MoE-sBOED-v2: read-only evaluation

Training was neither interrupted nor restarted. Process1209684 remains active. Evaluation used low-priority CPU only (2threads,nice19),168.71seconds; the GPU compute process remained the training worker. No model source, optimizer, training protocol, checkpoint-selection policy, or official randomness was modified.

Matched frozen checkpoint update: **1600**. Current live durable training count: **8192**, state **DEVELOPMENT_COMPLETE**. Four checkpoint copies and SHA256 values are preserved in checkpoint_snapshot.json/checkpoints/. All383frozen source/configuration hashes matched before and after.

Evaluation uses the exact previously saved development draw identity: seed97402751,N128,L65536,horizon30. Deterministic reconstruction was verified against saved curve SHA256 identity2024d7e2b0a9dcc8868a9fc730857eefd1fc8bea4440d3306b28c48e6c832586; reconstructed arrays and raw scores are saved in this directory. No official seed1001 or reserved confirmation randomness was consumed.

| Model / frozen intervention | EIG nats | MC SE |
|---|---:|---:|
| v1 | 9.193565 | 0.135350 |
| v2 | 8.920910 | 0.147232 |
| dense | 9.249552 | 0.155414 |
| shared | 9.238480 | 0.138216 |
| v2_routed_masked | 6.593657 | 0.179771 |
| v2_uniform_all7_mass_matched | 8.417578 | 0.162511 |
| v2_equal_selected_top3_weights | 8.930116 | 0.148875 |

Paired SE is sampleSD(per-case score difference)/sqrt128; it is not obtained by assuming the model scores independent.

| Paired comparison | Difference | SE | Approx95%MC interval |
|---|---:|---:|---|
| v2_minus_v1 | -0.272655 | 0.157748 | [-0.581841,0.036532] |
| v2_minus_dense | -0.328642 | 0.172420 | [-0.666585,0.009301] |
| v2_minus_trained_shared | -0.317570 | 0.166985 | [-0.644860,0.009720] |
| learned_minus_uniform_all7 | 0.503332 | 0.148080 | [0.213095,0.793569] |
| full_minus_routed_mask | 2.327254 | 0.208749 | [1.918106,2.736401] |
| learned_weights_minus_equal_top3 | -0.009205 | 0.036240 | [-0.080235,0.061824] |

**Competence test:** force each of seven experts into the weakest selected slot at stages1,10,20,29, preserving its original slot weight; score terminal full-horizon utility with paired truth/noise/contrast draws. Condition on observed prefix-mean response using independently fitted, pre-existing frozen thresholds. Exclude experts already occupying the first two slots. Stage1 is pooled. Apply Holm across all tested conditional effects, and separately across all context-by-expert interactions.

Conditional effects: **42** tested, **0** Holm significant. Expert/context interactions: **34** tested, **0** significant. No multiplicity-supported meaningful rank reversal. Counts before eligibility filtering: {'1': {'0': 128}, '10': {'0': 108, '1': 20}, '20': {'0': 71, '1': 57}, '29': {'0': 69, '1': 59}}. Minimum included bin count16; some comparisons were unavailable.

Strongest apparent interaction: stage20,expert0versus5, high-minus-low difference -0.062253±0.025652; Holm p=0.517852. Its point rank reversal does not survive multiplicity control.

**Conclusion:** v2 does not currently demonstrate higher EIG than matched native-v1, parameter-matched dense, or the separately trained shared-only control. Its routed component is causally important to its frozen deployed computation, and learned routing outperforms mass-matched uniform-all7 averaging. Neither fact establishes useful conditional specialization or an advantage over retrained controls. Equal weighting of the same learned top3 set performs indistinguishably from learned weights.

**Exact limitations:** one training seed202 and one intermediate1600-update snapshot;128development cases reused in prior learning curves, not untouched confirmation; finite-L sequential sPCE lower bound (ceiling log65537≈11.0904nats), not exact/infinite-contrast EIG or comparable to different-budget official references; MC SE conditions on the shared contrast bank and trained models and excludes training-seed/contrast-bank uncertainty; intervals use normal approximation; no independent competence replication in this read-only check. Holm counts apply to the stated test families, not all previous milestone looks. Contexts use a coarse observable summary and historical cutpoints (stage10only20high-context cases before eligibility), so power is limited. Weakest-slot substitutions test one constrained intervention, not all expert coalitions. Uniform-all7 changes support/computation from sparse3 to dense7 while preserving total routed mass, so it is not a pure weight-only test. Routed masking is an off-policy ablation without retraining; it is not the trained shared-only comparator. Lack of significant competence effects is not proof that every expert is useless. Training remains on its existing schedule; no new variant was launched.
