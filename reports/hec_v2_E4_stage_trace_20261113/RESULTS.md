# E4 preparation stage trace

**Finding: the short preparation never demonstrated improved heldout representation competency. The experiment does not support freezing the refreshed adapter to preserve a supposed gain.**

Prepared checkpoints were absent from the old run. Both short preparation phases were reproduced, saved separately and verified against their exact historical state digests and receipts. Original/final episode-level evaluations also reproduce. Zero new native training updates.

| Stage |128-coordinate probe MSE |52-coordinate adapter probe MSE |Diagnostic readout MSE |
|---|---:|---:|---:|
|original|0.170892955|0.163384496|0.624207908|
|ordinary_prepared|0.170892955|0.163384496|0.163438462|
|E4_prepared|0.171607807|0.163995446|0.179850815|
|ordinary_final|0.170657639|0.164423686|0.358652384|
|E4_final|0.171157768|0.164674197|0.350423388|

## Representation comparisons

```json
{
  "decision": "NO_DEMONSTRATED_PREPARATION_COMPETENCY_GAIN",
  "original_minus_prepared_error": {
    "n": 256,
    "mean": -0.0007148514156099901,
    "MCSE": 0.0005176539454095371,
    "CI95": [
      -0.0017342728018294817,
      0.0003045699706095016
    ],
    "p_two_sided": 0.16850503796159866
  },
  "final_minus_prepared_error": {
    "n": 256,
    "mean": -0.00045003922756058784,
    "MCSE": 0.0006683768773814343,
    "CI95": [
      -0.0017662808768311002,
      0.0008662024217099246
    ],
    "p_two_sided": 0.5013461895359254
  },
  "original_minus_final_error": {
    "n": 256,
    "mean": -0.00026481218804940223,
    "MCSE": 0.0007258463513087155,
    "CI95": [
      -0.0016942290797686212,
      0.001164604703669817
    ],
    "p_two_sided": 0.7155389844820756
  },
  "ordinary_final_minus_E4_final_error": {
    "n": 256,
    "mean": -0.0005001280726222815,
    "MCSE": 0.0003973310791631386,
    "CI95": [
      -0.0012825963665481078,
      0.0002823402213035449
    ],
    "p_two_sided": 0.20928307776566937
  }
}
```

## Optimization observations

Candidate full-calibration readout MSE:0.272795 after ridge calibration,0.820647 after first Adam step,0.286441 after300steps. This records a first-step overshoot and no net full-calibration readout improvement over initialization; it is not proof of an incorrect gradient or intrinsic expert incapacity.

Preparation changes adapter weights by4.99% relativeL2 and common-history actions byRMS0.0003573. Thus updates reach the deployed policy, but the measured target representation does not improve reliably. The unused readout improves through calibration; control head-only preparation achieves lower heldout readout error0.163438 than adapter+head0.179851. This is not representation specialization.

## Decision

Do not launch the frozen-refreshed-E4 native-training experiment. First establish a heldout representation improvement in a bounded preparation-only optimization/target-learnability study. These results locate the failed intervention before native continuation; they do not rule out other specialization methods.

Diagnostic replay of previous heldout bank,not independent new policy acceptance;model-derived teacher,one lineage. Different minibatch endpoint losses are not evidence of learning. Exact matching ensures preparation replay fidelity. No new long training or EIG optimization.

Artifacts: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v2_E4_stage_trace_20261113`
