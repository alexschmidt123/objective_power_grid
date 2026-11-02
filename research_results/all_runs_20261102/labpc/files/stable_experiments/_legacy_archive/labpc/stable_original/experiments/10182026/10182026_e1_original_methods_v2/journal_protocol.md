# E1 protocol

{
  "version": "E1_original_single_policy_20261018",
  "methods": [
    "random",
    "fixed",
    "dad",
    "rl_sboed",
    "vsoed",
    "scorebed",
    "moe",
    "resmoe",
    "switchmoe",
    "switchresmoe"
  ],
  "seeds": [
    101,
    102,
    103,
    104,
    105
  ],
  "budget": 1000000,
  "base_budget": 500000,
  "residual_budget": 500000,
  "development_seed": 1001,
  "consumed_confirmation_seed": 8831001,
  "new_confirmation_bank": "NOT_GENERATED; freeze after development resolution and analysis",
  "checkpoint_selection": "final_budget",
  "scorebed": {
    "restarts": 1,
    "initialization": 10014,
    "score": 692990,
    "policy": 296996,
    "final_params": "EMA"
  },
  "moe_training": {
    "moe": "straight_through",
    "resmoe": "relaxed",
    "switchmoe": "relaxed",
    "switchresmoe": "relaxed"
  },
  "code_commit": "5af7d0bba7def3a8e1948e6d8539d14791fc0d5d",
  "training_primary_accounting": "all pretraining plus newly generated training trajectories; validation/evaluation likelihood work separate",
  "analysis": {
    "paired_training_seeds": true,
    "seed_confidence_interval": "Student t95 with4df",
    "trajectory_confidence_interval": "paired MC; not independent training replicates",
    "principal_pairs": [
      [
        "resmoe",
        "scorebed"
      ],
      [
        "resmoe",
        "dad"
      ],
      [
        "resmoe",
        "moe"
      ],
      [
        "switchmoe",
        "moe"
      ],
      [
        "switchresmoe",
        "resmoe"
      ],
      [
        "switchresmoe",
        "switchmoe"
      ]
    ],
    "multiplicity": "Bonferroni simultaneous95% across6 contrasts for directional claims"
  },
  "sir": "held until location complete and SIR-specific native-baseline gates pass"
}

All corrected full runs use run.sh through existing tools.journal_campaign.run_phase. Code5af7d0b; primary episodes1M; no favorable seed or checkpoint replacement. A24h method-specific safety cap records incomplete work as partial, not as1M complete. Historical final8831001 cannot select corrections.
