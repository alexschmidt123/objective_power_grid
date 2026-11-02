# MoE-sBOED-v2 development pilot

{
  "protocol": {
    "candidate": "MoE-sBOED-v2 residual-context design readout",
    "source_base": "4b2bdfceca488f3b42be9d84c015dc8f9cf87a65",
    "audit_from": "/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10272026/10272026_moe_v2_recovered_audit",
    "training_seed": 202,
    "arms": [
      "v1",
      "v2",
      "dense",
      "shared"
    ],
    "maximum_matched_updates": 8192,
    "maximum_charged_training_seconds_per_arm": 7200,
    "save_reserve_seconds": 15,
    "batch_size": 2000,
    "training_contrasts": 2000,
    "horizon": 30,
    "optimizer": {
      "name": "Adam",
      "lr": 5e-05,
      "betas": [
        0.8,
        0.998
      ],
      "decay": 0.98,
      "decay_after_updates": [
        1,
        1001,
        2001,
        3001,
        4001,
        5001,
        6001,
        7001,
        8001
      ],
      "clip": 5
    },
    "balance_factor": 0.001,
    "curve_updates": [
      128,
      512,
      2048,
      4096,
      8192
    ],
    "curve_seed": 97402751,
    "curve_N": 128,
    "curve_L": 65536,
    "final_development_seed": 97402752,
    "final_replication_seed": 97402753,
    "final_N": 512,
    "final_L": 262144,
    "context_fit_seed": 97402754,
    "substitution_N": 256,
    "substitution_L": 65536,
    "maximum_diagnostic_seconds": 1800,
    "checkpoint_selection": "latest common completed update, never best EIG",
    "harmful_stop": "At or beyond4096matched updates, stop only if candidate-v1 paired upper95CI<-.1",
    "official_seed1001_used": false,
    "LabPC_source_modified": false,
    "no_full_official_training": true,
    "campaign": "/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10282026/10282026_moe_v2_context_readout_seed202"
  },
  "current_state": {
    "state": "DEVELOPMENT_COMPLETE",
    "pid": 1209684,
    "matched_updates": 8192,
    "charged_training_seconds": {
      "v1": 5235.493719878403,
      "v2": 5606.758765669336,
      "dense": 3626.8374811442627,
      "shared": 3883.133011688973
    },
    "stop_reason": "PREDECLARED_HARMFUL_REGRESSION_STOP",
    "diagnostic_GPU_seconds": 104.27807500225026,
    "results": [
      {
        "split": "development",
        "seed": 97402752,
        "EIG": {
          "v1": {
            "mean": 10.604669849747777,
            "MC_SE": 0.07501053296954512,
            "N": 512,
            "CI95": [
              10.457649205127469,
              10.751690494368086
            ]
          },
          "v2": {
            "mean": 9.566319767029302,
            "MC_SE": 0.0818290438472965,
            "N": 512,
            "CI95": [
              9.405934841088602,
              9.726704692970003
            ]
          },
          "dense": {
            "mean": 10.203234080558365,
            "MC_SE": 0.075604127232472,
            "N": 512,
            "CI95": [
              10.05504999118272,
              10.35141816993401
            ]
          },
          "shared": {
            "mean": 10.431250075070093,
            "MC_SE": 0.0810030958781087,
            "N": 512,
            "CI95": [
              10.272484007149,
              10.590016142991185
            ]
          },
          "masked": {
            "mean": 7.543817663673233,
            "MC_SE": 0.09816011672588473,
            "N": 512,
            "CI95": [
              7.3514238348904986,
              7.736211492455967
            ]
          },
          "uniform": {
            "mean": 9.268491438694191,
            "MC_SE": 0.08153267689148708,
            "N": 512,
            "CI95": [
              9.108687391986876,
              9.428295485401506
            ]
          }
        },
        "v2_minus_v1": {
          "mean": -1.038350082718474,
          "MC_SE": 0.10020771309075448,
          "N": 512,
          "CI95": [
            -1.2347572003763527,
            -0.8419429650605952
          ]
        },
        "v2_minus_dense": {
          "mean": -0.636914313529063,
          "MC_SE": 0.09492508927472582,
          "N": 512,
          "CI95": [
            -0.8229674885075257,
            -0.4508611385506004
          ]
        },
        "v2_minus_shared": {
          "mean": -0.8649303080407894,
          "MC_SE": 0.10097674800332168,
          "N": 512,
          "CI95": [
            -1.0628447341273,
            -0.6670158819542789
          ]
        },
        "full_minus_mask": {
          "mean": 2.022502103356069,
          "MC_SE": 0.10946926139448437,
          "N": 512,
          "CI95": [
            1.8079423510228794,
            2.2370618556892583
          ]
        },
        "learned_minus_uniform": {
          "mean": 0.29782832833511164,
          "MC_SE": 0.06845974712124542,
          "N": 512,
          "CI95": [
            0.1636472239774706,
            0.43200943269275266
          ]
        }
      },
      {
        "split": "replication",
        "seed": 97402753,
        "EIG": {
          "v1": {
            "mean": 10.57539710544171,
            "MC_SE": 0.0768493468333539,
            "N": 512,
            "CI95": [
              10.424772385648335,
              10.726021825235083
            ]
          },
          "v2": {
            "mean": 9.681534345234756,
            "MC_SE": 0.08264719127864652,
            "N": 512,
            "CI95": [
              9.519545850328608,
              9.843522840140903
            ]
          },
          "dense": {
            "mean": 10.321612428674394,
            "MC_SE": 0.07241263575505971,
            "N": 512,
            "CI95": [
              10.179683662594476,
              10.463541194754312
            ]
          },
          "shared": {
            "mean": 10.393354058383977,
            "MC_SE": 0.083060555959066,
            "N": 512,
            "CI95": [
              10.230555368704207,
              10.556152748063747
            ]
          },
          "masked": {
            "mean": 7.757381744888351,
            "MC_SE": 0.09211282193676018,
            "N": 512,
            "CI95": [
              7.5768406138923,
              7.937922875884401
            ]
          },
          "uniform": {
            "mean": 9.288479964617025,
            "MC_SE": 0.08570730358472839,
            "N": 512,
            "CI95": [
              9.120493649590957,
              9.456466279643093
            ]
          }
        },
        "v2_minus_v1": {
          "mean": -0.8938627602069527,
          "MC_SE": 0.09936017981114988,
          "N": 512,
          "CI95": [
            -1.0886087126368065,
            -0.6991168077770988
          ]
        },
        "v2_minus_dense": {
          "mean": -0.6400780834396376,
          "MC_SE": 0.0965755324734968,
          "N": 512,
          "CI95": [
            -0.8293661270876913,
            -0.45079003979158383
          ]
        },
        "v2_minus_shared": {
          "mean": -0.7118197131492211,
          "MC_SE": 0.10279100361204856,
          "N": 512,
          "CI95": [
            -0.9132900802288363,
            -0.5103493460696059
          ]
        },
        "full_minus_mask": {
          "mean": 1.924152600346406,
          "MC_SE": 0.10548637988341922,
          "N": 512,
          "CI95": [
            1.7173992957749045,
            2.1309059049179075
          ]
        },
        "learned_minus_uniform": {
          "mean": 0.3930543806177307,
          "MC_SE": 0.07856881830700745,
          "N": 512,
          "CI95": [
            0.23905949673599608,
            0.5470492644994653
          ]
        }
      }
    ]
  }
}

Actual progress/status, charged per-arm cost and checkpoint hashes are recorded in the referenced scratch campaign. Pending EIG cells are unavailable, never zero. Selection is the latest common completed update, never the largest test EIG.
