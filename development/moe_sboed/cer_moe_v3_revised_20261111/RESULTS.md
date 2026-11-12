# Revised CER-MoE V3 results

Experiment status: COMPLETE.

One revised candidate and one matched trained uniform-router control. No architecture sweep, extra reward or cost penalty. All previous artifacts are preserved.

## Training and exposure

```json
{
  "preparation": {
    "status": "COMPLETE",
    "updates": 200,
    "seconds": 29.94843245111406,
    "supported_history_examples": 35,
    "anchor_training_rows": [
      177,
      135,
      176,
      102
    ],
    "no_groups_required": true,
    "overlapping_weight_floor": 0.25,
    "effective_sample_size": [
      7752.60302734375,
      7424.0234375,
      7582.1953125,
      7442.24853515625
    ],
    "critic_rows": 8000000,
    "new_simulator_observations": 0,
    "audit_sha256": "26784e4e0aa865e079adfe26dae576e95b0e15905fbb4f32eae6d3cc54cf7206",
    "actor_sha256": "cc8ea3ac91edace9c285a8c63744f23f91e418a0e714c35f594699ae7aa9fccd",
    "objective": "native remaining variational Q on overlapping conditional replay distributions"
  },
  "arms": {
    "full": {
      "status": "COMPLETE",
      "arm": "full",
      "updates": 2000,
      "seconds": 4466.845335938968,
      "process_CPU_seconds": 4436.282054900999,
      "process_peak_RSS_bytes": 2684768256,
      "optimizer_steps": {
        "actor": 2000,
        "critic": 10000,
        "posterior": 300000
      },
      "simulator_counts": {
        "episodes": 2000000,
        "observations": 60000000,
        "batch_calls": 60000
      },
      "selected_counts": [
        15007545,
        14992479,
        15000888,
        14999088
      ],
      "model_queries": {
        "frozen_belief_rows": 1218041472
      },
      "GPU_peak_allocated_bytes": 2721006080,
      "actor_sha256": "bc7d527fc4c05077a06260687b0ffbf3a7e5f1bcbe2709f1e43dddd2b38bc048",
      "runtime_sha256": "397719bb149a840fa5e484d579c1c599dfb6a96060eb69d82f7087cdc4a0d2c1",
      "final_epsilon": 0.09999999999999998,
      "frozen_belief_unchanged": true
    },
    "control": {
      "status": "COMPLETE",
      "arm": "control",
      "updates": 2000,
      "seconds": 4425.723582822131,
      "process_CPU_seconds": 4391.234491946001,
      "process_peak_RSS_bytes": 2708668416,
      "optimizer_steps": {
        "actor": 2000,
        "critic": 10000,
        "posterior": 300000
      },
      "simulator_counts": {
        "episodes": 2000000,
        "observations": 60000000,
        "batch_calls": 60000
      },
      "selected_counts": [
        15008420,
        14991954,
        15000836,
        14998790
      ],
      "model_queries": {
        "frozen_belief_rows": 1218041472
      },
      "GPU_peak_allocated_bytes": 2721004032,
      "actor_sha256": "68e96798a927694eb64dca25b8d0ce487404720f61c7a3aa08583b7b1f45f2fe",
      "runtime_sha256": "fb86186890ca91d9d6beacf303b74b4d42a9e66f7fdaf056726ad86f02d5e680",
      "final_epsilon": 1.0,
      "frozen_belief_unchanged": true
    }
  }
}
```

## Independent common-bank full-horizon EIG

2000 trajectories, T30, 1M prior contrasts plus positive once. Finite-contrast lower bound; MCSE conditional on this one training lineage. Changed policies resimulated whole trajectories.

|Policy|EIG nats|MCSE|
|---|---:|---:|
|full|12.867752|0.037328|
|control|12.935021|0.034894|
|uniform|12.859814|0.037745|
|expert1|12.866084|0.037234|
|expert2|12.872856|0.036958|
|expert3|12.878269|0.037295|
|expert4|12.858548|0.037992|
|hec_v2|12.874375|0.036423|
|vsoed|12.426710|0.042573|
|dad|10.244953|0.048731|
|rl_sboed|11.176544|0.040225|
|scorebed|11.027039|0.046031|

Primary paired differences: candidate minus each comparison. Ordinary95% intervals are unadjusted; Holm7 p-values account for the primary family.

```json
{
  "control": {
    "n": 2000,
    "mean": -0.06726897369881309,
    "MCSE": 0.02424453878639662,
    "CI95": [
      -0.11481618537113508,
      -0.019721762026491098
    ],
    "p_two_sided": 0.005578310660504719,
    "Holm7_p": 0.039048174623533036
  },
  "uniform": {
    "n": 2000,
    "mean": 0.007938133247589987,
    "MCSE": 0.006023422062185919,
    "CI95": [
      -0.00387470948395262,
      0.01975097597913259
    ],
    "p_two_sided": 0.18769561587628267,
    "Holm7_p": 1.0
  },
  "expert1": {
    "n": 2000,
    "mean": 0.0016682848808928447,
    "MCSE": 0.01679587294867975,
    "CI95": [
      -0.03127096520541476,
      0.03460753496720045
    ],
    "p_two_sided": 0.9208885436880248,
    "Holm7_p": 1.0
  },
  "expert2": {
    "n": 2000,
    "mean": -0.005103713446955451,
    "MCSE": 0.014775926991093319,
    "CI95": [
      -0.034081543649874646,
      0.02387411675596374
    ],
    "p_two_sided": 0.7298245117243296,
    "Holm7_p": 1.0
  },
  "expert3": {
    "n": 2000,
    "mean": -0.010517275713264542,
    "MCSE": 0.01646887857434371,
    "CI95": [
      -0.04281524031795524,
      0.021780688891426155
    ],
    "p_two_sided": 0.5231464667436084,
    "Holm7_p": 1.0
  },
  "expert4": {
    "n": 2000,
    "mean": 0.009203521016178608,
    "MCSE": 0.015988763741990816,
    "CI95": [
      -0.022152865703198257,
      0.04055990773555547
    ],
    "p_two_sided": 0.5649339394645306,
    "Holm7_p": 1.0
  },
  "hec_v2": {
    "n": 2000,
    "mean": -0.006622838759254151,
    "MCSE": 0.03551125397364156,
    "CI95": [
      -0.07626578492169879,
      0.06302010740319049
    ],
    "p_two_sided": 0.8520718260590537,
    "Holm7_p": 1.0
  }
}
```

## Assessment

- Higher EIG than trained uniform-router control: not established.
- Higher EIG than uniform deployment of candidate: not established.
- Higher EIG than frozen HEC-v2: not established.
- Higher EIG than every frozen candidate expert: not established.
- Useful specialization requires conditional strength and routing evidence below; slot differences or nonzero exposure alone are insufficient.
- ICML-level novelty and training-seed robustness are not established by one pilot.

## Conditional competence and routing

```json
{
  "expected_routing_minus_uniform_MC": {
    "n": 128,
    "mean": -0.00017214150189204247,
    "MCSE": 0.00030007356876831124,
    "CI95": [
      -0.000765932941618777,
      0.00042164993783469203
    ],
    "p_two_sided": 0.5672094825582252
  },
  "critic_argmax_matches_MC_fraction": 0.2630208333333333,
  "MC_estimated_expert_means": [
    3.2390124797821045,
    3.2308294773101807,
    3.2327921390533447,
    3.2289083003997803
  ],
  "conditional_strengths": {
    "stage3_lower_uncertainty": {
      "calibration_best_expert": 2,
      "validation_cases": 29,
      "expert_minus_uniform": [
        {
          "n": 29,
          "mean": 0.01183483518403152,
          "MCSE": 0.06199106481339245,
          "CI95": [
            -0.11514810470721352,
            0.13881777507527657
          ],
          "p_two_sided": 0.8499721341494083
        },
        {
          "n": 29,
          "mean": -0.01565613417789854,
          "MCSE": 0.05324652659125034,
          "CI95": [
            -0.1247266995232062,
            0.0934144311674091
          ],
          "p_two_sided": 0.7709028333157288
        },
        {
          "n": 29,
          "mean": 0.07565890509506752,
          "MCSE": 0.06287836866159087,
          "CI95": [
            -0.05314159433576955,
            0.2044594045259046
          ],
          "p_two_sided": 0.23895184638839503
        },
        {
          "n": 29,
          "mean": -0.0718382638076256,
          "MCSE": 0.05873943212262447,
          "CI95": [
            -0.19216053607260664,
            0.04848400845735541
          ],
          "p_two_sided": 0.23152415757175804
        }
      ],
      "calibrated_expert_minus_others": {
        "n": 29,
        "mean": -0.020874590709291655,
        "MCSE": 0.0709953725779337,
        "CI95": [
          -0.16630201893234536,
          0.12455283751376206
        ],
        "p_two_sided": 0.770905559894333
      }
    },
    "stage3_higher_uncertainty": {
      "calibration_best_expert": 4,
      "validation_cases": 35,
      "expert_minus_uniform": [
        {
          "n": 35,
          "mean": 0.018310274396623884,
          "MCSE": 0.04971316829661356,
          "CI95": [
            -0.0827190389149567,
            0.11933958770820448
          ],
          "p_two_sided": 0.7149192151534411
        },
        {
          "n": 35,
          "mean": -0.021742514201572964,
          "MCSE": 0.05057646258735589,
          "CI95": [
            -0.12452625259543998,
            0.08104122419229406
          ],
          "p_two_sided": 0.6699852677673906
        },
        {
          "n": 35,
          "mean": 0.14471962792532786,
          "MCSE": 0.06413559893956995,
          "CI95": [
            0.014380409128583516,
            0.2750588467220722
          ],
          "p_two_sided": 0.030583657809285444
        },
        {
          "n": 35,
          "mean": -0.14128732681274414,
          "MCSE": 0.06853490663766229,
          "CI95": [
            -0.2805670145237358,
            -0.002007639101752473
          ],
          "p_two_sided": 0.04696349856435473
        }
      ],
      "calibrated_expert_minus_others": {
        "n": 35,
        "mean": -0.18838307516915456,
        "MCSE": 0.09137986939737325,
        "CI95": [
          -0.37408931301413656,
          -0.002676837324172565
        ],
        "p_two_sided": 0.04696351470980674
      }
    },
    "stage12_lower_uncertainty": {
      "calibration_best_expert": 1,
      "validation_cases": 32,
      "expert_minus_uniform": [
        {
          "n": 32,
          "mean": 0.01225777342915535,
          "MCSE": 0.04763747612823578,
          "CI95": [
            -0.08489949968676944,
            0.10941504654508014
          ],
          "p_two_sided": 0.7986369884906684
        },
        {
          "n": 32,
          "mean": -0.0802352987229824,
          "MCSE": 0.11229706688650755,
          "CI95": [
            -0.3092666766288914,
            0.14879607918292656
          ],
          "p_two_sided": 0.4802693867100456
        },
        {
          "n": 32,
          "mean": -0.015485964715480804,
          "MCSE": 0.059244315629385653,
          "CI95": [
            -0.1363155430641657,
            0.10534361363320412
          ],
          "p_two_sided": 0.7955194305865781
        },
        {
          "n": 32,
          "mean": 0.08346346020698547,
          "MCSE": 0.04862453766499518,
          "CI95": [
            -0.015706938185580804,
            0.18263385859955175
          ],
          "p_two_sided": 0.09605091888062615
        }
      ],
      "calibrated_expert_minus_others": {
        "n": 32,
        "mean": 0.01634371653199196,
        "MCSE": 0.06351663675760005,
        "CI95": [
          -0.1131993182050097,
          0.14588675126899361
        ],
        "p_two_sided": 0.7986367701265681
      }
    },
    "stage12_higher_uncertainty": {
      "calibration_best_expert": 1,
      "validation_cases": 32,
      "expert_minus_uniform": [
        {
          "n": 32,
          "mean": -0.032142868265509605,
          "MCSE": 0.04584694280045093,
          "CI95": [
            -0.12564832458319627,
            0.061362588052177064
          ],
          "p_two_sided": 0.4884761082278637
        },
        {
          "n": 32,
          "mean": 0.014325973577797413,
          "MCSE": 0.047399235883559966,
          "CI95": [
            -0.08234540535563828,
            0.1109973525112331
          ],
          "p_two_sided": 0.7644872558920487
        },
        {
          "n": 32,
          "mean": -0.028658160008490086,
          "MCSE": 0.04418901875195705,
          "CI95": [
            -0.11878225793616952,
            0.06146593791918935
          ],
          "p_two_sided": 0.5214140177316801
        },
        {
          "n": 32,
          "mean": 0.04647513013333082,
          "MCSE": 0.03917005425520542,
          "CI95": [
            -0.03341272221623748,
            0.12636298248289912
          ],
          "p_two_sided": 0.24444015632799546
        }
      ],
      "calibrated_expert_minus_others": {
        "n": 32,
        "mean": -0.042857213877141476,
        "MCSE": 0.061129257382466086,
        "CI95": [
          -0.16753115627690796,
          0.081816728522625
        ],
        "p_two_sided": 0.488475544820611
      }
    },
    "stage24_lower_uncertainty": {
      "calibration_best_expert": 1,
      "validation_cases": 33,
      "expert_minus_uniform": [
        {
          "n": 33,
          "mean": 0.005078121511773629,
          "MCSE": 0.01948306208525881,
          "CI95": [
            -0.03460757728239334,
            0.044763820305940595
          ],
          "p_two_sided": 0.7960379197692234
        },
        {
          "n": 33,
          "mean": -0.007681301428061543,
          "MCSE": 0.019170303295397515,
          "CI95": [
            -0.04672993141469981,
            0.03136732855857672
          ],
          "p_two_sided": 0.6913130033152092
        },
        {
          "n": 33,
          "mean": -0.00018599477003921163,
          "MCSE": 0.021191464667303306,
          "CI95": [
            -0.04335159574762595,
            0.04297960620754752
          ],
          "p_two_sided": 0.993051649855889
        },
        {
          "n": 33,
          "mean": 0.0027890651853698673,
          "MCSE": 0.026257312016721785,
          "CI95": [
            -0.05069532917112634,
            0.05627345954186607
          ],
          "p_two_sided": 0.9160705007905446
        }
      ],
      "calibrated_expert_minus_others": {
        "n": 33,
        "mean": 0.006770911880514838,
        "MCSE": 0.02597740914994372,
        "CI95": [
          -0.04614333899371106,
          0.059685162754740735
        ],
        "p_two_sided": 0.7960354180341911
      }
    },
    "stage24_higher_uncertainty": {
      "calibration_best_expert": 3,
      "validation_cases": 31,
      "expert_minus_uniform": [
        {
          "n": 31,
          "mean": -0.009384062021009384,
          "MCSE": 0.018156735108329793,
          "CI95": [
            -0.046465062029108994,
            0.027696937987090223
          ],
          "p_two_sided": 0.6090598803692866
        },
        {
          "n": 31,
          "mean": -0.016979481964822737,
          "MCSE": 0.019321119923352382,
          "CI95": [
            -0.05643847300917838,
            0.022479509079532913
          ],
          "p_two_sided": 0.386492265088747
        },
        {
          "n": 31,
          "mean": 0.001491996970388197,
          "MCSE": 0.022524959119969313,
          "CI95": [
            -0.0445101066196365,
            0.047494100560412894
          ],
          "p_two_sided": 0.9476282004806065
        },
        {
          "n": 31,
          "mean": 0.024871581684677832,
          "MCSE": 0.023520988661463002,
          "CI95": [
            -0.023164685603601777,
            0.07290784897295743
          ],
          "p_two_sided": 0.29876040501930196
        }
      ],
      "calibrated_expert_minus_others": {
        "n": 31,
        "mean": 0.0019893182381506887,
        "MCSE": 0.030033280157656016,
        "CI95": [
          -0.05934682260020869,
          0.06332545907651006
        ],
        "p_two_sided": 0.9476284934149339
      }
    }
  },
  "mean_conditional_MCSE": 0.8845340609550476,
  "diagnostic_initial_observations": 3840,
  "diagnostic_suffix_observations": 417792,
  "posterior_ESS_quantiles": [
    1.4435055255889893,
    25.12307224273682,
    739.1568908691406,
    2502.7836181640623,
    3162.466064453125
  ],
  "limitations": "SNIS and frozen variational posterior approximate conditional information; noisy MC maxima; one trained lineage; descriptive subgroup diagnostics",
  "selection_frequency": [
    15015,
    14903,
    15024,
    15058
  ],
  "stage_selection_frequency": [
    [
      527,
      476,
      491,
      506
    ],
    [
      513,
      523,
      469,
      495
    ],
    [
      460,
      541,
      492,
      507
    ],
    [
      526,
      511,
      459,
      504
    ],
    [
      472,
      540,
      485,
      503
    ],
    [
      522,
      462,
      520,
      496
    ],
    [
      518,
      505,
      500,
      477
    ],
    [
      500,
      502,
      488,
      510
    ],
    [
      506,
      474,
      494,
      526
    ],
    [
      498,
      519,
      486,
      497
    ],
    [
      511,
      494,
      481,
      514
    ],
    [
      479,
      503,
      500,
      518
    ],
    [
      515,
      475,
      507,
      503
    ],
    [
      492,
      457,
      528,
      523
    ],
    [
      511,
      488,
      513,
      488
    ],
    [
      492,
      496,
      509,
      503
    ],
    [
      526,
      506,
      508,
      460
    ],
    [
      500,
      511,
      462,
      527
    ],
    [
      470,
      503,
      518,
      509
    ],
    [
      517,
      479,
      502,
      502
    ],
    [
      508,
      487,
      527,
      478
    ],
    [
      531,
      484,
      489,
      496
    ],
    [
      521,
      500,
      475,
      504
    ],
    [
      505,
      517,
      519,
      459
    ],
    [
      488,
      490,
      499,
      523
    ],
    [
      479,
      499,
      541,
      481
    ],
    [
      459,
      513,
      513,
      515
    ],
    [
      510,
      494,
      505,
      491
    ],
    [
      464,
      470,
      531,
      535
    ],
    [
      495,
      484,
      513,
      508
    ]
  ],
  "within_stage_probability_SD": [
    0.003221024526283145,
    0.002200543647632003,
    0.0027561287861317396,
    0.0021836182568222284
  ],
  "mean_probability": [
    0.24989467859268188,
    0.24998347461223602,
    0.25007373094558716,
    0.2500465214252472
  ]
}
```

Conditional values use exact likelihoods with finite SNIS particles and a frozen normalized variational posterior, not exact conditional MI. Subgroup probes are descriptive and noisy, not additional rewards.

## Costs and limitations

Training receipts above contain actual simulator calls, optimizer steps, CPU time, peak RAM/VRAM and wall time. TRAINING_LOGS/resources.jsonl records actual GPU/CPU utilization, device memory and power every 10 seconds. Two disposable full native integration executions add 2000 trajectories/60000 observations; their optimizer steps do not initialize either scientific arm. A disposable preparation check adds 128 critic rows without new simulator episodes. Scientific preparation uses replay/value queries without new simulator episodes.

Evaluation receipts record resimulation and positive/negative likelihood terms. Conditional diagnostics add128 initial T30 rollouts and128x4x16x(27+18+6) future observations. Inference latency includes Bayesian information, four proposals, native critic and categorical selection. Frozen external models have different historical training/provider budgets.

```json
{
  "resource_summary": {
    "samples": 932,
    "CPU_percent_one_core": {
      "mean": 98.1929473577557,
      "maximum": 100.06372118013469
    },
    "RSS_bytes": {
      "mean": 2143546776.72103,
      "maximum": 2214998016.0
    },
    "GPU_percent": {
      "mean": 58.78004291845494,
      "maximum": 100.0
    },
    "GPU_memory_MiB": {
      "mean": 3795.9903433476393,
      "maximum": 3943.0
    },
    "GPU_power_W": {
      "mean": 147.25511802575093,
      "maximum": 197.3
    }
  },
  "evaluation_completion": {
    "status": "COMPLETE",
    "seconds": 419.72127399011515,
    "policies_completed": [
      "full",
      "control",
      "uniform",
      "expert1",
      "expert2",
      "expert3",
      "expert4",
      "hec_v2",
      "vsoed",
      "dad",
      "rl_sboed",
      "scorebed"
    ],
    "checkpoints_unchanged": true
  },
  "latency": {
    "seconds_per_design": {
      "full": 0.0009219797700643539,
      "control": 0.0008992454991675913
    },
    "repeats": 100,
    "history_stage": 12,
    "includes_belief_four_proposals_critic_and_selection": true
  },
  "preservation": {
    "status": "PASS",
    "files": 368
  }
}
```

No final-test bank, post-hoc training extension, independent specialization/diversity reward or computational-cost objective was used. A selection floor avoids zero exposure but cannot guarantee retained or useful competencies. Negative and incomplete results remain labeled.
