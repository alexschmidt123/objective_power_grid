# Pretraining correctness

Frozen-v1 diagnostics completed with no training or new rollouts. V2_CORE_NATIVE_CHECKS passes: initial candidate action matches parent exactly; trained stage architecture's commands/gates are bitwise invariant across different same-stage histories; analytic Jacobian and cross-hypothesis normal KL match independent calculations; gradients reach all legacy experts, new E2/E4 adapters, commander, coordinator and response maps; inherited Adam is preserved; the full10000×30 actor/target path succeeds with all32 fixture optimizer calls intercepted and only4 fixture episodes.

Native posterior allocation inherited an implicit device default; the new teacher now sets the actual device explicitly. The disposable gradient fixture was corrected to leave its no-grad reference block before backward. All failed logs are retained; no assertion was relaxed and no model was trained during these checks.

Before lock also require independent unrolled predictive-Bayes feature agreement, terminal-horizon handling, prefix packing/no future leakage, full-horizon reward telescoping, frozen-evaluator literal agreement, repeated counterfactual consistency and all available baseline-loader checks. Authoritative PASS receipts are in the runtime evidence directory. Protocol lock refuses missing/failed receipts.
