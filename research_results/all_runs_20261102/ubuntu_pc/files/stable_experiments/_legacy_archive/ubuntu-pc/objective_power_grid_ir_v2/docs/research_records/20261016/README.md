# Location-finding replication and troubleshooting records

The original ten-method location study completed training seeds101–105, with final1M-episode checkpoints and development evaluation1001 (2000 trajectories,L1M). The main CSVs are copied from generated campaign artifacts, not manually transcribed. Random has one shared reference and no training replicates. Across-seed SD is separate from trajectory MC uncertainty. Final8831001 remains unused. The six primary paired contrasts and adjusted intervals are in location_finding_comparisons.csv.

## Latest authorized runs

Only original-setting MoE103, SwitchMoE103 and ScoreBED102 are being repeated, in that order,1M episodes each. These are selected-seed reproducibility diagnostics, not a new independent five-seed study. Original results remain retained; no favorable-repeat replacement is authorized. SIR is deferred. No completed selected-repeat result was available at this sync snapshot. Live execution status remains in the LabPC campaign rather than this static Git record.

## Source changes

The active source change is the ScoreBED intermediate-validation bounds metadata fix. Native ScoreBED training, architecture and objectives are unchanged. Gamma remains0.001 with the established30% warmup/95% endpoint. The failed0.01 gamma candidate is archived as patches and a negative result; it is not promoted as a corrected method. Its MoE102 EIG8.15260 was worse than original9.72441.

## Evidence and limitations

The audit files are dated snapshots; some show only seeds101–104. Use location_finding_results.csv and location_finding_summary.csv for the completed five-seed table. ScoreBED's original diagnostic validation curves were unbounded and invalid; final held-out policy evaluation used correct bounds. The deeper score diagnostic finds approximation error and boundary concentration, but not a proven unique training defect. Do not claim variance is solved or temporal persistence is demonstrated. scorebed_training_diagnosis.md describes reference reliability exclusions.

## Reproduce

Portable fresh-training commands, with the same original settings:

```bash
bash run.sh --config configs/location_selected_seed_repeats.yaml --journal --variant moe --seed 103 --eval-seeds 1001 --eig-only
bash run.sh --config configs/location_selected_seed_repeats.yaml --journal --variant switchmoe --seed 103 --eval-seeds 1001 --eig-only
bash run.sh --config configs/location_selected_seed_repeats.yaml --journal --methods scorebed --seed 102 --eval-seeds 1001 --eig-only
```

Run each sequentially. These commands train full1M budgets and do not need old models. The separate score-matching environment is required for ScoreBED.

`tools/research_records/` contains the executed report/diagnosis and guarded sequential-runner logic. These archival scripts require the original dated experiment artifacts to replay comparisons; set MOE_PROJECT_ROOT to their repository. They intentionally refuse to overwrite existing campaigns. They do not ship pretrained weights, contrast banks, environments or trajectory dumps. `prepare_selected_rerun.py` creates a source-isolated copy and launches only the three declared jobs; do not invoke it while an existing queue is running. Use standalone root commands for a clean checkout without historical artifacts.

The native score-reference diagnostic is CPU-only, not a new policy-training objective. Full numerical arrays and exact campaign manifests stay under ignored experiments/. No AGENTS.md or private environment contents are published.

Figures: [instability](../../../documents/images/location_instability_20261016.png), [ScoreBED diagnosis](../../../documents/images/scorebed_diagnosis_20261016.png).
