# ScoreBED: troubleshooting report

This is a development troubleshooting report, not final confirmatory evidence. Training seeds 101–105 are running under the frozen fixed-budget protocol; only completed methods enter this snapshot. Evaluation uses seed 1001, 2,000 paired trajectories, L=1,000,000, and the final 1M-budget checkpoint. No retraining, checkpoint reselection, algorithm change, or reserved-seed evaluation was performed for this audit.

A nonzero gradient (>1e-8) is an exposure proxy, not proof of informative learning. Raw gradient norms are measured before component clipping; large raw values are not the applied parameter update. Correlation with sharpening does not establish the router as the sole cause.

## Assessment

Confirmed diagnostic validation defect; seed-102 final-policy weakness is real, underlying training cause unresolved.

`src/research/journal_scorebed_bridge.py::maybe_validate` saves `export_arrays(params)` without `common_design_bounds`. `OfficialScorePolicy` defaults to no bounding when this metadata is absent. By contrast, `scorebed_worker.py` includes bounds in the final official policy export. The final JAX-to-Torch bridge verifies bounded actions.

Consequence: intermediate validation scores evaluate the wrong policy mapping. They must not be used as valid learning curves, convergence evidence or comparisons. Final checkpoint selection is fixed-budget, so weights are not chosen using these erroneous scores. No final held-out result is replaced by an unbounded result. A small prior smoke test passing was insufficient to establish bounded validation equivalence.

Seed 102's boundary concentration and low EIG persist with the correct bounded final policy. The defect therefore does not establish why score-based training produced a weak seed. Frozen score-model quality, score-gradient direction on policy histories, and mismatch between score-training histories and policy histories remain untested.

Required correction after campaign completion: propagate bounds in intermediate exports; add action and complete-rollout parity tests between native JAX, validation and deployed Torch policies on identical histories and noise. Rescore saved candidates; missing historical candidates cannot be reconstructed from final weights. No full retraining is justified solely to repair diagnostic metadata. If the next audit finds a training-mechanism defect, declare it and rerun all affected seeds under one corrected configuration, retaining the originals.

## Measured evidence

| Seed | EIG | Trajectory MC SE | Boundary coordinates | Final-budget selection | Checkpoint finite / hash matches |
|---|---:|---:|---:|---|---|
| 101 | 9.91568 | 0.05223 | 0.00% | final | True / True |
| 102 | 7.18907 | 0.06226 | 28.29% | final_predeclared_fixed_budget | True / True |
| 103 | 10.13136 | 0.05285 | 0.00% | final_predeclared_fixed_budget | True / True |
| 104 | 9.79230 | 0.05206 | 0.00% | final_predeclared_fixed_budget | True / True |

Across 4 completed training seeds: mean 9.25710, sample SD 1.38579. This is an interim summary, not a five-seed result.

Bounded replay on the independent validation population (512, seed 771001; not final evaluation):

- Seed 102: incorrect logged 6.21599; corrected final bounded replay 7.18030.

- Seed 103: incorrect logged 6.42771; corrected final bounded replay 10.39889.

- Seed 104: incorrect logged 5.97182; corrected final bounded replay 9.72000.

## Reproducibility and limitations

See `evidence.json` for run paths, complete stage usage, validation history, gradients and checkpoint checks; `source_hashes.json` hashes inputs. Only saved complete runs are audited. The previous disk-full partial MoE-103 attempt remains archived and is not the accepted checkpoint. External GPU sharing affects wall-clock comparisons; no evidence identifies it as the cause of EIG degradation. No final seed 8831001 was used.
