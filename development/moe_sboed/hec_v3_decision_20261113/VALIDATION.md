# Actual-checkpoint V3 validation

LabPC validation output:scratch_runs/hec_v3_decision_20261113_validation02/VALIDATION.json.

- Independent expert parameter storage,standalone router bypass,exact selected-proposal execution and uniform sampling intervals pass CPU tests.
- Poisoned unobserved design/observation padding does not affect proposals/selection.
- Exact categorical expected-Q actor loss matches explicit sum; differentiated experts produce finite nonzero router gradients. Router-only phase leaves expert gradients absent; joint mode yields expert gradients. This is computational validation,not routing-quality evidence.
- Actual frozen V2:each expert exactly reproduces parent actions at B2,stages0,14,29. V3 initialization executes those same actions. Full30-stage rollout agrees with original V2 using the exact location-finding simulator/noise chronology.
- One full native update passes:actor1,critic5,posterior150; frozen experts preserved. Fresh validation episodes1000,observations30000,kept separate from pilot accounting. Earlier validation01 is preserved separately.
- Full adapted native runtime serializes/deserializes correctly with identical actor-state digest and callable categorical actor loss. No native vendor file edits; guarded AST finds exactlyone actor block.
- Verified parameter counts:593917 total,24451 router; three189822-parameter complete experts.

No claim of specialization,training convergence or performance improvement follows from these tests. The expert warm-up and selector pilot must be evaluated independently.
