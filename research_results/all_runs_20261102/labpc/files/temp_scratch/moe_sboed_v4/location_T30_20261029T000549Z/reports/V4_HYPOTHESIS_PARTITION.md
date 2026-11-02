# Frozen prior-predictive partition

Domain adapter src/research/moe_v4_partition.py: sample8192 prior parameters; predict noise-free observation means on a fixed5x5 grid over the existing feasible location box; standardize by sqrt(prior predictive variance+observation noise variance); kmeans++ initialization and deterministic Lloyd updates into4groups. No held-out EIG chooses the grid, partition, or thresholds. A separate4096-prior validation set checks every group>=5%,between variance>=5% and within-group variance>=1% of total. Source-label exchange leaves signatures and labels unchanged. Groups are domain-dependent predictive hypotheses, not physical parameter-coordinate rules or named strategies.

Exactly the SAME four information roles apply to every partition. E1 does not belong to group0; it predicts discrimination among ALL groups. SIR references are16 fixed increasing feasible times; grid references use the canonical configured feasible sequence with the existing physical backend and required configuration metadata. Distinct domain partitions and weights are expected, not zero-shot transfer.

Actual independent validation:

{
  "valid": true,
  "groups": 4,
  "fit_samples": 8192,
  "validation_samples": 4096,
  "seed": 86404011,
  "validation_seed": 86404012,
  "frequency": [
    0.29150390625,
    0.208740234375,
    0.28466796875,
    0.215087890625
  ],
  "within_variance_relative_to_total": [
    0.5610588711119504,
    0.6590753230060054,
    0.5533729054143418,
    0.6626856012060182
  ],
  "between_fraction": 0.3988104209830498,
  "iterations": 27,
  "reference_designs": [
    [
      -4.0,
      -4.0
    ],
    [
      -4.0,
      -2.0
    ],
    [
      -4.0,
      0.0
    ],
    [
      -4.0,
      2.0
    ],
    [
      -4.0,
      4.0
    ],
    [
      -2.0,
      -4.0
    ],
    [
      -2.0,
      -2.0
    ],
    [
      -2.0,
      0.0
    ],
    [
      -2.0,
      2.0
    ],
    [
      -2.0,
      4.0
    ],
    [
      0.0,
      -4.0
    ],
    [
      0.0,
      -2.0
    ],
    [
      0.0,
      0.0
    ],
    [
      0.0,
      2.0
    ],
    [
      0.0,
      4.0
    ],
    [
      2.0,
      -4.0
    ],
    [
      2.0,
      -2.0
    ],
    [
      2.0,
      0.0
    ],
    [
      2.0,
      2.0
    ],
    [
      2.0,
      4.0
    ],
    [
      4.0,
      -4.0
    ],
    [
      4.0,
      -2.0
    ],
    [
      4.0,
      0.0
    ],
    [
      4.0,
      2.0
    ],
    [
      4.0,
      4.0
    ]
  ],
  "method": "standardized predictive mean signatures; frozen kmeans++/Lloyd; noise-floor scaling",
  "labels_are_not_physical_strategies": true,
  "source_exchangeability": true,
  "CPU_preparation_seconds": 0.05290705797960982,
  "partition_sha256": "bfb29be1532f33c0646580d3818f7ca4129f6b9d8e81a823492ddfeef305bc00"
}
