# Fresh corrected V2 specialization audit

Confirmed diagnostic issue: old evaluate_v2.py:89 removed response coordinates0:12 from every expert, dropping learned E1/E3 coordinates. Corrected128-coordinate probe masks only E2/E4 analytic prefixes. Frozen actor code/checkpoints remain unchanged.

640 fresh episodes:256 calibration,128 rival selection,256 independent test. Targets are existing model-derived tasks,not physical ground truth or exact EIG.

| Expert | Relative MSE reduction vs constant | Designated advantage95%CI |
|---|---:|---|
|E1|79.05%|[-0.005593035825238936, 0.028355079306335823]|
|E2|83.76%|[0.028429580891929606, 0.06350157791912732]|
|E3|71.08%|[0.2130793276117193, 0.2726320957096124]|
|E4|19.94%|[-0.00655450196634475, 0.0140104353265174]|

All expert packets have nonzero direct-fusion action sensitivity. Old E2/E4 diagnostic readouts underperform constants, but those readouts are absent from the deployed action computation.

E4 probe MSE0.198640 versus estimated target MC noise0.039044; excess0.159596,95%CI[0.114027,0.205165]. This supports testing an E4 adapter refresh using its existing task; it does not establish a network bug.

Chosen next intervention: candidate refreshes existing adapter4 plus its diagnostic head; control refreshes only the unused diagnostic head. Both get matched preparation data/steps and2000 native EIG updates. Acceptance requires independent fresh competency and EIG gains. Full locked protocol is development/moe_sboed/hec_v2_specialization_audit_20261113/REPAIR_PROTOCOL.md.

One checkpoint. Probes assess decodability of existing model-derived targets, not EIG or scientific accuracy. E4 labels use current V2 posterior/critic and16 conditional replicates; comparison with historical V1-teacher diagnostics is not numerically matched. Diagnostic readouts are not in the action graph.
