# Frozen uniform101 routing diagnosis

Completed in142.26 seconds. No model training or optimizer updates;all checkpoint files preserved.25 histories at stages1,10,20,30;14 weight choices/history;32 suffix replicates split16 discovery/16 held-out validation. Same exact likelihood,physical bounds,T30 and1M-contrast scorer;fresh diagnostic draws,not the primary benchmark bank.

| Choice at frozen history | Held-out EIG difference from uniform | 95% CI across source-trajectory clusters |
|---|---:|---:|
| Weights selected on discovery simulations | +0.00710 | [-0.04074,+0.05494] |
| Weights selected by native critic | -0.01271 | [-0.07875,+0.05334] |
| Transferred learned101 router weights | -0.01536 | [-0.07415,+0.04344] |

Intervals use7 complete trajectory clusters,averaging stages10/20/30 within a trajectory. They are descriptive single-seed diagnostic intervals,not training-seed uncertainty or exact-posterior guarantees. Initial stage is reported separately in raw results because its history is identical across episodes.

Maximum action change across tested weights ranges0.217–2.324 physical units per history (median0.441). Gates affect final designs. Mean critic Spearman rank correlation is0.277,median0.382;ranking measurements themselves have Monte Carlo error.

0 positive individual history/weight cells survive a Bonferroni correction across325 inspected nonuniform cells. Per-history-only intervals in raw rows do not protect the whole search.

## Judgment

The current test finds action sensitivity but does not demonstrate a reliably exploitable EIG improvement over uniform. It therefore cannot distinguish limited useful reweighting headroom from noisy/inaccurate critic guidance. It does not establish poor expert complementarity,and it does not establish router optimization failure. Large harmful weight changes are not evidence of available positive routing gains.

Do not promote or retrain the prepared continuation pair on the claim that this diagnostic has proved a routing fix. The next evidence-gathering step would be more held-out suffix replication at the same prespecified histories,with posterior/saturation checks. The previously requested seed101 trial implementation is saved and CPU-tested but has not started GPU training;the latest request to diagnose frozen weights takes precedence.

## Limitations

- Conditional theta draws use approximate finite-pool self-normalized importance resampling;ESS thresholds are diagnostic,not a proof of posterior accuracy.
- Held-out suffix validation limits candidate-selection optimism;shared prior contrast-bank uncertainty and posterior approximation error are excluded from reported CIs.
- Transferred router comes from the separately learned101 checkpoint and different learned expert representations;transfer failure alone does not prove a router-learning bug.
- One-step intervention then uniform continuation;not unrestricted sequence-level routing optimization.
- This is a fresh diagnostic bank,not the2000-case/1M-contrast primary benchmark. Do not report these conditional EIGs as final method EIG.
- Only7 complete source-trajectory clusters;one late-stage posterior ESS23.69 failed64 threshold,excluding its cluster from complete-case pooled inference. Selection of complete cases can affect generalization.
- The exploratory raw report pools23 noninitial histories as though independent. Use this clustered summary instead;histories at three stages share original trajectories.
- Estimator ceiling saturation remains in some conditional cases despite1M contrasts;this can hide weight differences.
- Native critic estimates the variational actor-critic return,while this diagnostic scores the benchmark prior-contrast EIG estimator. Ranking disagreement can include bound/estimator mismatch;it is not automatically a critic implementation bug.
- No independently trained seed101 continuation pair is complete in this new trial. Earlier benchmark results remain unchanged.

## Reproduction and artifacts

`tools/moe_sboed_frozen_routing_diagnostic_seed101.py` executed frozen rollouts;`configs/moe_sboed_frozen_routing_diagnostic_seed101_20261116.json` locks the full diagnostic. Raw per-history results,paired suffix score arrays,parameter/noise draws and prefixes remain under`experiments/location_finding/moe_sboed_uniform_residual_seed101_20261116/frozen_diagnostic/full/`. Source/protocol hashes and executed commit are recorded in`frozen_diagnostic/FULL_LAUNCH.json`. Run`tools/moe_sboed_frozen_routing_diagnostic_summary.py` on the saved outputs to reproduce this clustered report without GPU computation.
