# V3 decision-level MoE pilot

Accepted under locked rule:**False**

Three complete V2-derived policies,independent action heads,categorical selector;uniform and standalone comparisons use identical physical execution. No role-specific specialization claim.

```json
{
  "status": "COMPLETE",
  "primary_learned_minus_control": {
    "n": 1024,
    "mean": -0.03832347111007516,
    "MCSE": 0.05240981332581228,
    "CI95": [
      -0.14116649412771434,
      0.06451955190756403
    ],
    "p_two_sided": 0.4648079556505861
  },
  "secondary_learned_minus_original": {
    "n": 1024,
    "mean": -0.07044651945063336,
    "MCSE": 0.0475246166032529,
    "CI95": [
      -0.1637033911904463,
      0.02281035228917959
    ],
    "p_two_sided": 0.1385640899643741
  },
  "internal_comparisons": {
    "uniform": {
      "n": 1024,
      "mean": 0.029608901647003926,
      "MCSE": 0.03987696547346227,
      "CI95": [
        -0.04864109425674169,
        0.10785889755074955
      ],
      "p_two_sided": 0.45795103485656347,
      "p_Holm4": 1
    },
    "expert1": {
      "n": 1024,
      "mean": -0.0033481720050185793,
      "MCSE": 0.04575942404344515,
      "CI95": [
        -0.09314123176881244,
        0.08644488775877528
      ],
      "p_two_sided": 0.9416859256111146,
      "p_Holm4": 1
    },
    "expert2": {
      "n": 1024,
      "mean": -0.05122267285195625,
      "MCSE": 0.045472000476248386,
      "CI95": [
        -0.14045172548309218,
        0.03800637977917967
      ],
      "p_two_sided": 0.2602323993899297,
      "p_Holm4": 1
    },
    "expert3": {
      "n": 1024,
      "mean": 0.0064803401812678265,
      "MCSE": 0.028362483982993913,
      "CI95": [
        -0.049174954276037096,
        0.06213563463857275
      ],
      "p_two_sided": 0.8193165777089353,
      "p_Holm4": 1
    }
  },
  "means": {
    "original": 14.030370819089327,
    "control": 13.998247770748769,
    "learned": 13.959924299638695,
    "uniform": 13.930315397991691,
    "expert1": 13.963272471643712,
    "expert2": 14.01114697249065,
    "expert3": 13.953443959457427
  },
  "selection_counts": {
    "original": [
      0,
      0,
      0
    ],
    "control": [
      0,
      0,
      0
    ],
    "learned": [
      3544,
      3311,
      23865
    ],
    "uniform": [
      10233,
      10275,
      10212
    ],
    "expert1": [
      30720,
      0,
      0
    ],
    "expert2": [
      0,
      30720,
      0
    ],
    "expert3": [
      0,
      0,
      30720
    ]
  },
  "limitations": "Short pilot,one lineage,finite16M prior-sPCE,episode/simulator budget matched but compute/parameters differ. Independent native-EIG training does not guarantee expert specialization or named strategy roles.",
  "accepted": false
}
```

Artifacts:`/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_decision_20261113_pilot`
