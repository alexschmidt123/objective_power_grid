# Actual-checkpoint validation

Pathway validation:scratch_runs/hec_v3_v2_pathways_20261113_validation/VALIDATION.json. B2 at stages0,14,29.

- All FOUR first/response/legacy packets match corresponding frozen V2 computations within float tolerances. Own-head proposals match V2 one-hot pathway designs.
- Exported standalone policies match their proposals and contain no other expert's query modules.
- Initial routing weights match V2. Uniform sampling chooses the correct hard proposals,not a coordinate average. Unobserved history padding cannot affect decisions.
- Router gradients finite/nonzero;standalone head and E4 adapter have native actor-loss gradients.
- Full native selector update passes:actor1,critic5,posterior150,1000 validation episodes/30000 observations. Context/expert actor weights preserved. Guarded actor update changes exactly one native AST block,not vendor files.
- Verified236296 total parameters and25956 routing parameters.

Preparation/standalone validation:scratch_runs/hec_v3_v2_pathways_20261113_preparation_validation02/VALIDATION.json. Uses the completed preparation from validation01,which was preserved after a missing-import error in the generic logger was corrected. No historical model was changed.

- All four heldout teacher-action errors improve after300 head-only steps:E1 0.104796->0.008711;E2 0.082299->0.008406;E3 0.115985->0.010974;E4 0.038926->0.007018. This is action imitation,not an EIG result.
- One complete standalone native update passes (in addition to the separate selector fixture),saving actor,target and full runtime.
- SHA-verified historical DAD,RL-sBOED and ScoreBED baseline adapters load and execute two full episodes with reduced31 contrasts in separate namespaces. These smoke scores are ceiling-limited correctness checks,not baseline performance results. vsOED uses the existing validated frozen actor loader/evaluator.

No quality win is inferred from successful tests. The pilot acceptance rule is in PROTOCOL.md.
