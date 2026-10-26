# FH-MoE-v3 independent context discovery

Outcome: FAILED predeclared discovery gate; zero stages passed all checks. No specialist or primary training was launched. This was a numerically successful discovery experiment, not a troubleshooting failure.

The router-independent shared-only reference and paired native-v1 control each completed 600 fresh optimizer updates, B=2000, L=2000, H=30. No pretrained model was used. Their parameters were initialized identically before routed masking in the reference. The reference was then frozen. Three independent splits each supplied 8000 complete trajectories; fitting, independent refitting and held-out checking used disjoint simulation seeds. No expert labels/exposure were used.

| Stage (remaining experiments) | Held-out gradient MSE difference +/- SE | Between-context cosine | Replicate assignment kappa | Minimum split mean norm | Minimum mean SNR | Gate |
|---|---:|---:|---:|---:|---:|---|
| 1 (29) | +0.000739 +/- 0.000743 | 0.901931 | 0.075045 | 0.023546 | 0.749249 | FAIL |
| 10 (20) | -0.007932 +/- 0.001703 | -0.911495 | 0.693064 | 0.059015 | 2.515431 | FAIL |
| 20 (10) | -0.012023 +/- 0.001910 | -0.252170 | 0.607813 | 0.053876 | 2.340036 | FAIL |
| 29 (1) | -0.017173 +/- 0.002047 | -0.862939 | 0.807957 | 0.043256 | 1.572445 | FAIL |

Stages 10,20,29 passed held-out prediction, direction separation and replicated assignment checks. Thus the data provide limited evidence of history-predictable gradient structure. Every stage failed the joint coherence/SNR requirements; stage1 also failed predictability, separation and assignment replication. Required minimum coherence was .1 and minimum SNR was3 in both halves of every context. No thresholds were changed after results. This is more informative than saying no mathematical clusters could be formed, and weaker than demonstrating useful specialists.

Full-graph precision probe: all stages passed. Float32 reference-network gradients differed from float64 by at most 0.00000458 relative norm; the simulator and likelihood remained canonical float64. Context-assignment agreement was100% in float32 arithmetic. This does not establish equivalence of all-float32 simulator training.

All raw adjoints, observable covariates and independent splits are retained as gradient_dataset_split0/1/2.pt under the dated development campaign. Summary checks and their exact results are in fh_moe_v3_evidence/context_gate.json. State-specific posterior information advantages were not estimated. Normal-approximation case-level intervals and heuristic thresholds do not establish population-wide or multiplicity-controlled confirmation.

Correctness:23 tests passed on UbuntuPC before the pilot (1.77s) and after completion (1.74s), including unchanged foundation, exact native gradients, full30-stage finite differences, live history gradients, reference independence from all routed weights, rejection of unpredictable synthetic labels, reproducibility, normalized soft responsibilities, checkpoint roundtrips and deployment with optimizer/autograd calls forbidden. LabPC integration repeats are recorded separately. Specialist gradient allocation and joint-training tests are not claimed: that conditional phase was not entered.
