# Expert specialization evidence

PENDING trained-policy diagnostics. Four training targets are mathematically distinct components, not yet four proven useful information-acquisition experts.

Held-out diagnostics: per-role squared error/R2/calibration bins, realized-target correlations, negative-value frequencies, full-history expert selection, physical next-action change under individual masks, and paired full-horizon EIG under masks. Learned versus uniform top2 preserves selected-softmax gate semantics and active count, with two random routing realizations. Controlled expert-function SWAPS at stages5,15,25 preserve two distinct expert functions, gate weights and the entire common prefix; the original policy resumes afterward.

Observable context definitions are predicted coarse-share>=.1 and predicted immediate-share>=1/(remaining decisions), conditional on a positive predicted total. They are fixed information-based definitions, not median-feature responsibilities or forced stage assignments. Each swap/context interaction needs>=16cases per cohort; missing coverage is reported rather than forced. Holm correction covers eligible tests in the prespecified36-test interaction family. Effects include nonlinear fusion/slot interactions, not additive expert EIG contributions. Replication across two independently trained seeds is required for a stable attribution pattern. Correlated targets, poor prediction calibration or unreplicated utility mean the four intended roles are not established even if EIG improves.

Role calibration also compares against cross-fitted stage-only target means to distinguish history learning from time trends. Frozen individual-expert mask effects use a separate four-test Holm family.


Measured role calibration:

{
  "101": {
    "role_metrics": [
      {
        "role": 0,
        "MSE": 0.06231756513796517,
        "cross_fitted_stage_only_baseline_MSE": 0.06203008805095495,
        "skill_vs_stage_baseline": -0.004634478138642484,
        "R2": 0.004228382633802097,
        "prediction_mean": 0.03639823643955929,
        "realized_mean": 0.03959340371790951,
        "negative_target_fraction": 0.3619791666666667,
        "negative_prediction_fraction": 0.0,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.022639577508107078,
            "realized_mean": 0.023839026761434512
          },
          {
            "N": 1536,
            "predicted_mean": 0.03160465583519156,
            "realized_mean": 0.02402206383385305
          },
          {
            "N": 1536,
            "predicted_mean": 0.036783691522554146,
            "realized_mean": 0.028389830874832583
          },
          {
            "N": 1536,
            "predicted_mean": 0.04183410626891986,
            "realized_mean": 0.0454378484120769
          },
          {
            "N": 1536,
            "predicted_mean": 0.0491291510630238,
            "realized_mean": 0.07627824870735049
          }
        ]
      },
      {
        "role": 1,
        "MSE": 0.4471493842766826,
        "cross_fitted_stage_only_baseline_MSE": 0.413884084685816,
        "skill_vs_stage_baseline": -0.0803734688569111,
        "R2": -0.005472660597541035,
        "prediction_mean": 0.24990172255834991,
        "realized_mean": 0.28041901736001684,
        "negative_target_fraction": 0.26419270833333336,
        "negative_prediction_fraction": 0.0,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.1759717851845041,
            "realized_mean": 0.29456402860798325
          },
          {
            "N": 1536,
            "predicted_mean": 0.22520731693121365,
            "realized_mean": 0.1924007855312216
          },
          {
            "N": 1536,
            "predicted_mean": 0.2562816480595142,
            "realized_mean": 0.26673848520187643
          },
          {
            "N": 1536,
            "predicted_mean": 0.28092485517884813,
            "realized_mean": 0.3298667369118166
          },
          {
            "N": 1536,
            "predicted_mean": 0.31112300743766946,
            "realized_mean": 0.31852505054718633
          }
        ]
      },
      {
        "role": 2,
        "MSE": 0.396266113662474,
        "cross_fitted_stage_only_baseline_MSE": 0.2938208542948035,
        "skill_vs_stage_baseline": -0.34866571882226793,
        "R2": -0.007180915860993897,
        "prediction_mean": 0.2701300959100758,
        "realized_mean": 0.35668560620606143,
        "negative_target_fraction": 0.14765625,
        "negative_prediction_fraction": 0.0,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.07596444168582074,
            "realized_mean": 0.3617235489032166
          },
          {
            "N": 1536,
            "predicted_mean": 0.18285690674156127,
            "realized_mean": 0.22162990408691
          },
          {
            "N": 1536,
            "predicted_mean": 0.26295829696867573,
            "realized_mean": 0.24990353914311048
          },
          {
            "N": 1536,
            "predicted_mean": 0.34771963177978593,
            "realized_mean": 0.32885608246360504
          },
          {
            "N": 1536,
            "predicted_mean": 0.48115120237453546,
            "realized_mean": 0.6213149564334652
          }
        ]
      },
      {
        "role": 3,
        "MSE": 7.163636144938389,
        "cross_fitted_stage_only_baseline_MSE": 2.96152099622616,
        "skill_vs_stage_baseline": -1.4189043920562936,
        "R2": 0.07509069272004387,
        "prediction_mean": 2.0827790261868047,
        "realized_mean": 2.6475930888701535,
        "negative_target_fraction": 0.09986979166666667,
        "negative_prediction_fraction": 0.0,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.6406225009574283,
            "realized_mean": 1.9713190620608136
          },
          {
            "N": 1536,
            "predicted_mean": 1.2105801558206333,
            "realized_mean": 1.9777320353173942
          },
          {
            "N": 1536,
            "predicted_mean": 1.8903165431480797,
            "realized_mean": 1.9922584920685906
          },
          {
            "N": 1536,
            "predicted_mean": 2.797092223529768,
            "realized_mean": 2.567577331046172
          },
          {
            "N": 1536,
            "predicted_mean": 3.875283707478115,
            "realized_mean": 4.729078523857796
          }
        ]
      }
    ],
    "target_correlations": [
      [
        1.0,
        -0.007742268132533839,
        -0.11619468268221175,
        0.07869665765404395
      ],
      [
        -0.007742268132533839,
        0.9999999999999999,
        0.1597208487883863,
        0.13098107402751305
      ],
      [
        -0.11619468268221177,
        0.15972084878838633,
        0.9999999999999998,
        0.3851501235326059
      ],
      [
        0.07869665765404395,
        0.13098107402751305,
        0.3851501235326059,
        1.0
      ]
    ],
    "mean_action_change_when_selected": [
      0.09878057804622091,
      0.06429278167956788,
      0.505723888336076,
      0.13866670366094058
    ],
    "expert_selected_slot_share": [
      0.21412760416666668,
      0.18932291666666667,
      0.39622395833333335,
      0.20032552083333333
    ],
    "mean_physical_next_action_change_when_masked": [
      0.0423032970304712,
      0.02434419389637804,
      0.4007598417204899,
      0.05555695926623882
    ],
    "seconds": 5.715935730957426,
    "cases_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
    "contrasts": 262144,
    "interpretation": "Prediction and action changes do not prove beneficial causal competence; correlations are descriptive, trajectory observations are dependent"
  },
  "102": {
    "role_metrics": [
      {
        "role": 0,
        "MSE": 0.0666411685376271,
        "cross_fitted_stage_only_baseline_MSE": 0.06495834980821452,
        "skill_vs_stage_baseline": -0.025906118834314373,
        "R2": 0.0031354624548595167,
        "prediction_mean": 0.048739215219189,
        "realized_mean": 0.03961257075375887,
        "negative_target_fraction": 0.35703125,
        "negative_prediction_fraction": 0.0,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.021611053083797658,
            "realized_mean": 0.053368373317100204
          },
          {
            "N": 1536,
            "predicted_mean": 0.0321517311032908,
            "realized_mean": 0.017526332866236113
          },
          {
            "N": 1536,
            "predicted_mean": 0.04349460705950897,
            "realized_mean": 0.019436901742460776
          },
          {
            "N": 1536,
            "predicted_mean": 0.057386963018420244,
            "realized_mean": 0.018374294479671007
          },
          {
            "N": 1536,
            "predicted_mean": 0.08905172183092731,
            "realized_mean": 0.08935695136332623
          }
        ]
      },
      {
        "role": 1,
        "MSE": 0.46253910061635534,
        "cross_fitted_stage_only_baseline_MSE": 0.4607085357290808,
        "skill_vs_stage_baseline": -0.003973368725147752,
        "R2": 0.00899608110911565,
        "prediction_mean": 0.30549692913030385,
        "realized_mean": 0.2960288024132763,
        "negative_target_fraction": 0.24895833333333334,
        "negative_prediction_fraction": 0.0,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.207804048007015,
            "realized_mean": 0.2263093559351316
          },
          {
            "N": 1536,
            "predicted_mean": 0.24904965354702166,
            "realized_mean": 0.2511771153825417
          },
          {
            "N": 1536,
            "predicted_mean": 0.28613691157693083,
            "realized_mean": 0.2686328664095388
          },
          {
            "N": 1536,
            "predicted_mean": 0.3451290880747609,
            "realized_mean": 0.3080553886749792
          },
          {
            "N": 1536,
            "predicted_mean": 0.43936494444579094,
            "realized_mean": 0.42596928566419007
          }
        ]
      },
      {
        "role": 2,
        "MSE": 0.3392457981499108,
        "cross_fitted_stage_only_baseline_MSE": 0.29674061386238576,
        "skill_vs_stage_baseline": -0.14324019801090304,
        "R2": 0.03857015570672451,
        "prediction_mean": 0.3174936330961033,
        "realized_mean": 0.2494715354613122,
        "negative_target_fraction": 0.153125,
        "negative_prediction_fraction": 0.0,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.07230000178918046,
            "realized_mean": 0.18475354874398522
          },
          {
            "N": 1536,
            "predicted_mean": 0.2081113445580368,
            "realized_mean": 0.07844428767126327
          },
          {
            "N": 1536,
            "predicted_mean": 0.2946813969623214,
            "realized_mean": 0.1875644786662974
          },
          {
            "N": 1536,
            "predicted_mean": 0.38345452924108026,
            "realized_mean": 0.25914632054434844
          },
          {
            "N": 1536,
            "predicted_mean": 0.6289208929298975,
            "realized_mean": 0.5374490416806665
          }
        ]
      },
      {
        "role": 3,
        "MSE": 6.195176950509419,
        "cross_fitted_stage_only_baseline_MSE": 3.265165953897403,
        "skill_vs_stage_baseline": -0.8973543881022845,
        "R2": 0.3611105563994179,
        "prediction_mean": 3.5614766236730455,
        "realized_mean": 3.505272546998175,
        "negative_target_fraction": 0.07044270833333334,
        "negative_prediction_fraction": 0.00026041666666666666,
        "calibration_bins": [
          {
            "N": 1536,
            "predicted_mean": 0.7456614449562363,
            "realized_mean": 1.9107959922157562
          },
          {
            "N": 1536,
            "predicted_mean": 2.27926183294862,
            "realized_mean": 1.466129423923724
          },
          {
            "N": 1536,
            "predicted_mean": 3.422950983748518,
            "realized_mean": 2.9343444352601584
          },
          {
            "N": 1536,
            "predicted_mean": 4.683554905884416,
            "realized_mean": 4.337503374947301
          },
          {
            "N": 1536,
            "predicted_mean": 6.675953950827439,
            "realized_mean": 6.877589508643933
          }
        ]
      }
    ],
    "target_correlations": [
      [
        1.0,
        -0.033232201959827394,
        -0.10764357568835373,
        0.1356945066568244
      ],
      [
        -0.03323220195982739,
        1.0,
        0.06133228632401155,
        0.029232495944791093
      ],
      [
        -0.10764357568835373,
        0.06133228632401155,
        1.0,
        0.29507861825272713
      ],
      [
        0.1356945066568244,
        0.029232495944791097,
        0.29507861825272713,
        1.0
      ]
    ],
    "mean_action_change_when_selected": [
      0.032882383063549646,
      0.11987703491494173,
      0.0752080758858483,
      0.20709928868852978
    ],
    "expert_selected_slot_share": [
      0.2825520833333333,
      0.29908854166666665,
      0.23209635416666666,
      0.18626302083333332
    ],
    "mean_physical_next_action_change_when_masked": [
      0.018581971679141298,
      0.0717076951040679,
      0.034911040433990695,
      0.07714987824712026
    ],
    "seconds": 5.688446020008996,
    "cases_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
    "contrasts": 262144,
    "interpretation": "Prediction and action changes do not prove beneficial causal competence; correlations are descriptive, trajectory observations are dependent"
  }
}

Measured causal interventions:

{
  "101": {
    "interventions": {
      "mask0": {
        "full_minus_intervention": {
          "mean": 0.019477237933941523,
          "MC_SE": 0.08333410828202237,
          "CI95": [
            -0.14385761429882232,
            0.18281209016670535
          ],
          "N": 256,
          "pvalue": 0.8151987251533745,
          "Holm_pvalue_four_mask_family": 1.0
        },
        "metadata": {
          "seed": 86404103,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
          "seconds": 2.6528087380575016,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask1": {
        "full_minus_intervention": {
          "mean": 0.00613740880738739,
          "MC_SE": 0.03805899831786401,
          "CI95": [
            -0.06845822789562607,
            0.08073304551040085
          ],
          "N": 256,
          "pvalue": 0.8718883212551707,
          "Holm_pvalue_four_mask_family": 1.0
        },
        "metadata": {
          "seed": 86404103,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
          "seconds": 2.6769182770513,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask2": {
        "full_minus_intervention": {
          "mean": 0.7013210851526789,
          "MC_SE": 0.16178243641591744,
          "CI95": [
            0.3842275097774807,
            1.018414660527877
          ],
          "N": 256,
          "pvalue": 1.4578378392460875e-05,
          "Holm_pvalue_four_mask_family": 5.83135135698435e-05
        },
        "metadata": {
          "seed": 86404103,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
          "seconds": 2.648995960946195,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask3": {
        "full_minus_intervention": {
          "mean": -0.03245750642812269,
          "MC_SE": 0.07762443334900594,
          "CI95": [
            -0.18460139579217433,
            0.11968638293592897
          ],
          "N": 256,
          "pvalue": 0.6758482846494127,
          "Holm_pvalue_four_mask_family": 1.0
        },
        "metadata": {
          "seed": 86404103,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
          "seconds": 2.6533689480274916,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "uniform86404991": {
        "full_minus_intervention": {
          "mean": 0.5160262110032336,
          "MC_SE": 0.1314055695908404,
          "CI95": [
            0.2584712946051865,
            0.7735811274012808
          ],
          "N": 256
        },
        "same_K": 2,
        "same_selected_softmax_semantics": true,
        "metadata": {
          "seed": 86404103,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
          "seconds": 2.77623783191666,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "uniform86404992": {
        "full_minus_intervention": {
          "mean": 0.4671881996428395,
          "MC_SE": 0.13429874995008179,
          "CI95": [
            0.20396264974067918,
            0.7304137495449998
          ],
          "N": 256
        },
        "same_K": 2,
        "same_selected_softmax_semantics": true,
        "metadata": {
          "seed": 86404103,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
          "seconds": 2.774720846908167,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask_all": {
        "full_minus_intervention": {
          "mean": 0.7361518281506946,
          "MC_SE": 0.16106654775771367,
          "CI95": [
            0.4204613945455758,
            1.0518422617558134
          ],
          "N": 256
        },
        "metadata": {
          "seed": 86404103,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "f13de5b365499b8d2f04453674613a0a7adfe907996e1a4e9cb36f4c2fc25ee3",
          "seconds": 2.326682596001774,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      }
    },
    "context_interactions": [
      {
        "stage": 5,
        "a": 1,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 1,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 2,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 2,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 1,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.031556924192819184,
          "MC_SE": 0.027587513350849555,
          "CI95": [
            -0.0856284503604843,
            0.022514601974845942
          ],
          "N": 191
        },
        "low": {
          "mean": -0.04252796504710379,
          "MC_SE": 0.025302265009802025,
          "CI95": [
            -0.09212040446631575,
            0.007064474372108179
          ],
          "N": 65
        },
        "interaction": 0.010971040854284603,
        "SE": 0.037433614673306054,
        "pvalue": 0.7694610634873177,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 1,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 2,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.02255033036150876,
          "MC_SE": 0.0510679141995287,
          "CI95": [
            -0.12264344219258501,
            0.07754278146956749
          ],
          "N": 191
        },
        "low": {
          "mean": -0.16059960990946895,
          "MC_SE": 0.07851055611754428,
          "CI95": [
            -0.3144802998998557,
            -0.006718919919082161
          ],
          "N": 65
        },
        "interaction": 0.1380492795479602,
        "SE": 0.09365809779499311,
        "pvalue": 0.14048951595323209,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 2,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 2,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.029796579746008293,
          "MC_SE": 0.05289769623147127,
          "CI95": [
            -0.07388290486767538,
            0.13347606435969198
          ],
          "N": 191
        },
        "low": {
          "mean": -0.19078252068431464,
          "MC_SE": 0.09307615485931091,
          "CI95": [
            -0.37321178420856405,
            -0.00835325716006527
          ],
          "N": 65
        },
        "interaction": 0.22057910043032294,
        "SE": 0.10705763340365522,
        "pvalue": 0.03936250064468168,
        "Holm_pvalue": 0.7085250116042702
      },
      {
        "stage": 15,
        "a": 2,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 3,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.023464991143680603,
          "MC_SE": 0.022398900498413088,
          "CI95": [
            -0.06736683612057026,
            0.02043685383320905
          ],
          "N": 191
        },
        "low": {
          "mean": -0.01095048543263167,
          "MC_SE": 0.01970586479156707,
          "CI95": [
            -0.049573980424103124,
            0.027673009558839785
          ],
          "N": 65
        },
        "interaction": -0.012514505711048934,
        "SE": 0.029833401594879065,
        "pvalue": 0.6748656032233108,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 3,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 3,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.024683589527816855,
          "MC_SE": 0.02195908781243891,
          "CI95": [
            -0.018356222584563407,
            0.06772340164019712
          ],
          "N": 191
        },
        "low": {
          "mean": 0.016286327402851807,
          "MC_SE": 0.01066001100776682,
          "CI95": [
            -0.00460729417237116,
            0.03717994897807478
          ],
          "N": 65
        },
        "interaction": 0.008397262124965048,
        "SE": 0.024409780257923522,
        "pvalue": 0.7308371137237759,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 3,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 3,
        "b": 2,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.0515479957645663,
          "MC_SE": 0.042699273460277735,
          "CI95": [
            -0.13523857174671067,
            0.03214258021757806
          ],
          "N": 191
        },
        "low": {
          "mean": -0.07964236881648727,
          "MC_SE": 0.029805580381578423,
          "CI95": [
            -0.13806130636438096,
            -0.021223431268593565
          ],
          "N": 65
        },
        "interaction": 0.028094373051920973,
        "SE": 0.05207303117659189,
        "pvalue": 0.5895290110553628,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 3,
        "b": 2,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 25,
        "a": 1,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.014238832975759385,
          "MC_SE": 0.0129946341712624,
          "CI95": [
            -0.03970831595143369,
            0.011230649999914919
          ],
          "N": 225
        },
        "low": {
          "mean": -0.006665476727865678,
          "MC_SE": 0.004421749998452759,
          "CI95": [
            -0.015332106724833084,
            0.002001153269101728
          ],
          "N": 31
        },
        "interaction": -0.007573356247893706,
        "SE": 0.013726339289619699,
        "pvalue": 0.5811272089091752,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 1,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": -0.0051221309599254325,
          "MC_SE": 0.019308674777485494,
          "CI95": [
            -0.042967133523797,
            0.03272287160394614
          ],
          "N": 17
        },
        "low": {
          "mean": -0.013904983145568891,
          "MC_SE": 0.012172132650258172,
          "CI95": [
            -0.03776236314007491,
            0.009952396848937126
          ],
          "N": 239
        },
        "interaction": 0.00878285218564346,
        "SE": 0.02282511193659706,
        "pvalue": 0.700393743428852,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.015625720412462407,
          "MC_SE": 0.02863566361375709,
          "CI95": [
            -0.040500180270501486,
            0.0717516210954263
          ],
          "N": 225
        },
        "low": {
          "mean": 0.04385072301528994,
          "MC_SE": 0.04659643176547381,
          "CI95": [
            -0.04747828324503873,
            0.1351797292756186
          ],
          "N": 31
        },
        "interaction": -0.02822500260282753,
        "SE": 0.054692126342598055,
        "pvalue": 0.6058050479689493,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": 0.09323326809437577,
          "MC_SE": 0.11985089042298246,
          "CI95": [
            -0.14167447713466985,
            0.3281410133234214
          ],
          "N": 17
        },
        "low": {
          "mean": 0.01376650187729557,
          "MC_SE": 0.026305681547897402,
          "CI95": [
            -0.03779263395658334,
            0.06532563771117447
          ],
          "N": 239
        },
        "interaction": 0.0794667662170802,
        "SE": 0.12270380930061274,
        "pvalue": 0.5172237711904967,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.019716261686677082,
          "MC_SE": 0.028963571423899305,
          "CI95": [
            -0.03705233830416556,
            0.07648486167751972
          ],
          "N": 225
        },
        "low": {
          "mean": 0.039586740336615425,
          "MC_SE": 0.04533306849883786,
          "CI95": [
            -0.04926607392110677,
            0.12843955459433762
          ],
          "N": 31
        },
        "interaction": -0.019870478649938343,
        "SE": 0.05379568355498089,
        "pvalue": 0.7118524129259788,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": 0.09022815538128146,
          "MC_SE": 0.11981351658081085,
          "CI95": [
            -0.14460633711710782,
            0.32506264787967076
          ],
          "N": 17
        },
        "low": {
          "mean": 0.017278113759228608,
          "MC_SE": 0.026588577828177643,
          "CI95": [
            -0.03483549878399957,
            0.0693917263024568
          ],
          "N": 239
        },
        "interaction": 0.07295004162205285,
        "SE": 0.12272828209661087,
        "pvalue": 0.5522426974625388,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.005003180081567591,
          "MC_SE": 0.009788641792118547,
          "CI95": [
            -0.024188917994119943,
            0.01418255783098476
          ],
          "N": 225
        },
        "low": {
          "mean": -0.007523243481936122,
          "MC_SE": 0.003628778816722027,
          "CI95": [
            -0.014635649962711293,
            -0.00041083700116094964
          ],
          "N": 31
        },
        "interaction": 0.002520063400368531,
        "SE": 0.010439614161217851,
        "pvalue": 0.8092495265572519,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": -0.0005292179017392292,
          "MC_SE": 0.0005292179017392292,
          "CI95": [
            -0.0015664849891481184,
            0.0005080491856696601
          ],
          "N": 17
        },
        "low": {
          "mean": -0.005648281849218246,
          "MC_SE": 0.009225415328357165,
          "CI95": [
            -0.02373009589279829,
            0.012433532194361798
          ],
          "N": 239
        },
        "interaction": 0.005119063947479016,
        "SE": 0.009240582209374505,
        "pvalue": 0.5795950355106947,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.00026009406174364833,
          "MC_SE": 0.008160390636137967,
          "CI95": [
            -0.01573427158508677,
            0.016254459708574063
          ],
          "N": 225
        },
        "low": {
          "mean": -0.007318170469930501,
          "MC_SE": 0.0034489996400747266,
          "CI95": [
            -0.014078209764476966,
            -0.0005581311753840371
          ],
          "N": 31
        },
        "interaction": 0.0075782645316741495,
        "SE": 0.008859321297458615,
        "pvalue": 0.39232967529120366,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": -0.0052835752486858675,
          "MC_SE": 0.025921801461560735,
          "CI95": [
            -0.05609030611334491,
            0.04552315561597317
          ],
          "N": 17
        },
        "low": {
          "mean": -0.0003285411776061294,
          "MC_SE": 0.00748368349633415,
          "CI95": [
            -0.014996560830421063,
            0.014339478475208804
          ],
          "N": 239
        },
        "interaction": -0.004955034071079738,
        "SE": 0.026980461628479904,
        "pvalue": 0.8542859336878169,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 2,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.027866656696233692,
          "MC_SE": 0.025314594975366752,
          "CI95": [
            -0.07748326284795252,
            0.02174994945548514
          ],
          "N": 225
        },
        "low": {
          "mean": 0.05188315183001713,
          "MC_SE": 0.033123592195519615,
          "CI95": [
            -0.013039088873201321,
            0.11680539253323557
          ],
          "N": 31
        },
        "interaction": -0.07974980852625083,
        "SE": 0.041689340108737045,
        "pvalue": 0.055753884443441586,
        "Holm_pvalue": 0.9478160355385069
      },
      {
        "stage": 25,
        "a": 3,
        "b": 2,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": 0.09999820240620567,
          "MC_SE": 0.12297235363828095,
          "CI95": [
            -0.141027610724825,
            0.3410240155372363
          ],
          "N": 17
        },
        "low": {
          "mean": -0.02661752925032446,
          "MC_SE": 0.02262748042889364,
          "CI95": [
            -0.070967390890956,
            0.01773233239030708
          ],
          "N": 239
        },
        "interaction": 0.12661573165653014,
        "SE": 0.1250368051011317,
        "pvalue": 0.31123802647427184,
        "Holm_pvalue": 1.0
      }
    ],
    "minimum_cohort": 16,
    "predeclared_family": 36,
    "interpretation": "Single-decision swaps share the complete prefix and preserve two distinct expert functions; downstream policy continues normally. Effects include nonlinear fusion interactions, not additive expert EIG."
  },
  "102": {
    "interventions": {
      "mask0": {
        "full_minus_intervention": {
          "mean": 0.07310020624281607,
          "MC_SE": 0.044223633325765975,
          "CI95": [
            -0.013578115075685238,
            0.15977852756131738
          ],
          "N": 256,
          "pvalue": 0.09833758109752239,
          "Holm_pvalue_four_mask_family": 0.29501274329256716
        },
        "metadata": {
          "seed": 86404113,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
          "seconds": 2.645291469991207,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask1": {
        "full_minus_intervention": {
          "mean": 0.2695356543725972,
          "MC_SE": 0.09047749374613806,
          "CI95": [
            0.0921997666301666,
            0.4468715421150278
          ],
          "N": 256,
          "pvalue": 0.0028915739458819697,
          "Holm_pvalue_four_mask_family": 0.011566295783527879
        },
        "metadata": {
          "seed": 86404113,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
          "seconds": 2.6459363100584596,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask2": {
        "full_minus_intervention": {
          "mean": -0.06921225839263045,
          "MC_SE": 0.055636668723118114,
          "CI95": [
            -0.17826012908994196,
            0.03983561230468105
          ],
          "N": 256,
          "pvalue": 0.21349795876685834,
          "Holm_pvalue_four_mask_family": 0.4269959175337167
        },
        "metadata": {
          "seed": 86404113,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
          "seconds": 2.6504757270449772,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask3": {
        "full_minus_intervention": {
          "mean": -0.0003338123643100571,
          "MC_SE": 0.10235018888061934,
          "CI95": [
            -0.20094018257032398,
            0.20027255784170384
          ],
          "N": 256,
          "pvalue": 0.997397725764753,
          "Holm_pvalue_four_mask_family": 0.997397725764753
        },
        "metadata": {
          "seed": 86404113,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
          "seconds": 2.6538001489825547,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "uniform86404991": {
        "full_minus_intervention": {
          "mean": 0.6063360102712474,
          "MC_SE": 0.11248086724098161,
          "CI95": [
            0.3858735104789235,
            0.8267985100635713
          ],
          "N": 256
        },
        "same_K": 2,
        "same_selected_softmax_semantics": true,
        "metadata": {
          "seed": 86404113,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
          "seconds": 2.77235369104892,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "uniform86404992": {
        "full_minus_intervention": {
          "mean": 0.5483716656772386,
          "MC_SE": 0.11661629738438263,
          "CI95": [
            0.31980372280384867,
            0.7769396085506286
          ],
          "N": 256
        },
        "same_K": 2,
        "same_selected_softmax_semantics": true,
        "metadata": {
          "seed": 86404113,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
          "seconds": 2.775125863030553,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      },
      "mask_all": {
        "full_minus_intervention": {
          "mean": 0.3742535829029623,
          "MC_SE": 0.10581945310119853,
          "CI95": [
            0.16684745482461316,
            0.5816597109813114
          ],
          "N": 256
        },
        "metadata": {
          "seed": 86404113,
          "systems": 256,
          "contrasts": 262144,
          "paired_draws_sha256": "d039278faa0d70b90e93a77ff514bb17df7541a4ea0b3e87967958f4a882a5a4",
          "seconds": 2.3266993379220366,
          "forward_particle_stages": 2013273600,
          "device": "cuda:0",
          "mode": "hard_deployed_policy",
          "secondary_parameter_inference": false
        }
      }
    },
    "context_interactions": [
      {
        "stage": 5,
        "a": 1,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 15,
        "low_N": 241,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 1,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 15,
        "low_N": 241,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 15,
        "low_N": 241,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 2,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 15,
        "low_N": 241,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 15,
        "low_N": 241,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 2,
        "context": "coarse_share_at_least_point1",
        "tested": false,
        "high_N": 15,
        "low_N": 241,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 5,
        "a": 3,
        "b": 2,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 256,
        "low_N": 0,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 1,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.015860885846553117,
          "MC_SE": 0.026246285571384448,
          "CI95": [
            -0.0355818338733604,
            0.06730360556646664
          ],
          "N": 74
        },
        "low": {
          "mean": 0.03713663202545308,
          "MC_SE": 0.029776116619182048,
          "CI95": [
            -0.02122455654814373,
            0.09549782059904989
          ],
          "N": 182
        },
        "interaction": -0.021275746178899962,
        "SE": 0.03969237492534042,
        "pvalue": 0.5919475249188302,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 1,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 254,
        "low_N": 2,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 2,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.028979059616375025,
          "MC_SE": 0.018716907006082136,
          "CI95": [
            -0.06566419734829601,
            0.007706078115545963
          ],
          "N": 74
        },
        "low": {
          "mean": -0.02665793776951214,
          "MC_SE": 0.053033081344575184,
          "CI95": [
            -0.1306027772048795,
            0.07728690166585522
          ],
          "N": 182
        },
        "interaction": -0.0023211218468628843,
        "SE": 0.05623904626480302,
        "pvalue": 0.9670787156005918,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 2,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 254,
        "low_N": 2,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 2,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.05946543584995602,
          "MC_SE": 0.11657575921413331,
          "CI95": [
            -0.16902305220974528,
            0.2879539239096573
          ],
          "N": 74
        },
        "low": {
          "mean": 0.10251685226294609,
          "MC_SE": 0.06843211082937788,
          "CI95": [
            -0.03161008496263455,
            0.2366437894885267
          ],
          "N": 182
        },
        "interaction": -0.04305141641299007,
        "SE": 0.13517714832365654,
        "pvalue": 0.7501197510722479,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 2,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 254,
        "low_N": 2,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 3,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.031205693556198753,
          "MC_SE": 0.06102289300582376,
          "CI95": [
            -0.15081056384761332,
            0.0883991767352158
          ],
          "N": 74
        },
        "low": {
          "mean": 0.06561866387277002,
          "MC_SE": 0.0394651341989649,
          "CI95": [
            -0.01173299915720119,
            0.14297032690274122
          ],
          "N": 182
        },
        "interaction": -0.09682435742896878,
        "SE": 0.07267248645906181,
        "pvalue": 0.182748954954029,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 3,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 254,
        "low_N": 2,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 3,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.12555554445422265,
          "MC_SE": 0.11104280439110933,
          "CI95": [
            -0.09208835215235164,
            0.3431994410607969
          ],
          "N": 74
        },
        "low": {
          "mean": 0.07422161626298723,
          "MC_SE": 0.06761217518338822,
          "CI95": [
            -0.058298247096453676,
            0.20674147962242814
          ],
          "N": 182
        },
        "interaction": 0.05133392819123542,
        "SE": 0.13000734840797018,
        "pvalue": 0.6929506021864004,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 3,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 254,
        "low_N": 2,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 15,
        "a": 3,
        "b": 2,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.07017785369403022,
          "MC_SE": 0.063785808813137,
          "CI95": [
            -0.05484233157971831,
            0.19519803896777876
          ],
          "N": 74
        },
        "low": {
          "mean": -0.003430785506613657,
          "MC_SE": 0.02215151877524421,
          "CI95": [
            -0.04684776230609231,
            0.03998619129286499
          ],
          "N": 182
        },
        "interaction": 0.07360863920064388,
        "SE": 0.06752273091334549,
        "pvalue": 0.27565533141675513,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 15,
        "a": 3,
        "b": 2,
        "context": "immediate_above_average_remaining_step",
        "tested": false,
        "high_N": 254,
        "low_N": 2,
        "reason": "Insufficient observable-context coverage; no forced median split"
      },
      {
        "stage": 25,
        "a": 1,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": 0.010428962609596424,
          "MC_SE": 0.010185074219250901,
          "CI95": [
            -0.00953378286013534,
            0.030391708079328186
          ],
          "N": 101
        },
        "low": {
          "mean": 0.04093385999193891,
          "MC_SE": 0.02792557136785573,
          "CI95": [
            -0.013800259889058322,
            0.09566797987293614
          ],
          "N": 155
        },
        "interaction": -0.030504897382342487,
        "SE": 0.02972496043854143,
        "pvalue": 0.30477920071491293,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 1,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": 0.012297153021914711,
          "MC_SE": 0.007700168139813439,
          "CI95": [
            -0.0027951765321196295,
            0.027389482575949052
          ],
          "N": 68
        },
        "low": {
          "mean": 0.034903548493774306,
          "MC_SE": 0.023505324971246622,
          "CI95": [
            -0.011166888449869072,
            0.08097398543741768
          ],
          "N": 188
        },
        "interaction": -0.022606395471859594,
        "SE": 0.024734447464726354,
        "pvalue": 0.3607357426769899,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.026578032417011198,
          "MC_SE": 0.02920490664484218,
          "CI95": [
            -0.08381964944090187,
            0.03066358460687947
          ],
          "N": 101
        },
        "low": {
          "mean": 0.029153406716496112,
          "MC_SE": 0.036722639305225055,
          "CI95": [
            -0.04282296632174499,
            0.10112977975473722
          ],
          "N": 155
        },
        "interaction": -0.05573143913350731,
        "SE": 0.04691991911412047,
        "pvalue": 0.2349125569989164,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": -0.008791025879735107,
          "MC_SE": 0.02537202677016633,
          "CI95": [
            -0.05852019834926111,
            0.0409381465897909
          ],
          "N": 68
        },
        "low": {
          "mean": 0.012937162376386988,
          "MC_SE": 0.03289341930464222,
          "CI95": [
            -0.05153393946071177,
            0.07740826421348575
          ],
          "N": 188
        },
        "interaction": -0.021728188256122095,
        "SE": 0.04154174738714112,
        "pvalue": 0.6009431896908936,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.09267219357930571,
          "MC_SE": 0.07190877707544104,
          "CI95": [
            -0.23361339664717012,
            0.04826900948855871
          ],
          "N": 101
        },
        "low": {
          "mean": 0.06428591290054844,
          "MC_SE": 0.05815457555996739,
          "CI95": [
            -0.04969705519698764,
            0.17826888099808452
          ],
          "N": 155
        },
        "interaction": -0.15695810647985414,
        "SE": 0.09248149479244716,
        "pvalue": 0.08966189294828543,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 2,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": 0.030232370641668387,
          "MC_SE": 0.05919771880388752,
          "CI95": [
            -0.08579515821395114,
            0.14625989949728793
          ],
          "N": 68
        },
        "low": {
          "mean": -0.00772008646573574,
          "MC_SE": 0.058038481712855826,
          "CI95": [
            -0.12147551062293316,
            0.10603533769146167
          ],
          "N": 188
        },
        "interaction": 0.03795245710740413,
        "SE": 0.08290256492484198,
        "pvalue": 0.64709905754504,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 0,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.03973102045476663,
          "MC_SE": 0.024433309517873314,
          "CI95": [
            -0.08762030710979832,
            0.008158266200265059
          ],
          "N": 101
        },
        "low": {
          "mean": -0.013869450422595127,
          "MC_SE": 0.015073288123129613,
          "CI95": [
            -0.04341309514392917,
            0.015674194298738914
          ],
          "N": 155
        },
        "interaction": -0.025861570032171505,
        "SE": 0.028708720431936337,
        "pvalue": 0.36768068862243963,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 0,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": -0.0013492251148788219,
          "MC_SE": 0.004301057852436763,
          "CI95": [
            -0.009779298505654878,
            0.007080848275897234
          ],
          "N": 68
        },
        "low": {
          "mean": -0.03229175837032933,
          "MC_SE": 0.017977219218996258,
          "CI95": [
            -0.067527108039562,
            0.002943591298903328
          ],
          "N": 188
        },
        "interaction": 0.030942533255450513,
        "SE": 0.018484574907145047,
        "pvalue": 0.09413745881604318,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 1,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.06347757631418639,
          "MC_SE": 0.05573124458310591,
          "CI95": [
            -0.17271081569707397,
            0.045755663068701186
          ],
          "N": 101
        },
        "low": {
          "mean": 0.058029125197431974,
          "MC_SE": 0.0561631108714857,
          "CI95": [
            -0.052050572110679996,
            0.16810882250554393
          ],
          "N": 155
        },
        "interaction": -0.12150670151161837,
        "SE": 0.0791218468284504,
        "pvalue": 0.12461420381258334,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 1,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": 0.016276229154301793,
          "MC_SE": 0.020492816239702016,
          "CI95": [
            -0.02388969067551416,
            0.05644214898411774
          ],
          "N": 68
        },
        "low": {
          "mean": 0.00785370008179047,
          "MC_SE": 0.05481931800120234,
          "CI95": [
            -0.09959216320056612,
            0.11529956336414705
          ],
          "N": 188
        },
        "interaction": 0.008422529072511323,
        "SE": 0.05852446619620842,
        "pvalue": 0.885567853550908,
        "Holm_pvalue": 1.0
      },
      {
        "stage": 25,
        "a": 3,
        "b": 2,
        "context": "coarse_share_at_least_point1",
        "tested": true,
        "high": {
          "mean": -0.0656251845571002,
          "MC_SE": 0.03375520072409532,
          "CI95": [
            -0.13178537797632703,
            0.0005350088621266236
          ],
          "N": 101
        },
        "low": {
          "mean": 0.0023781576597788363,
          "MC_SE": 0.004976123699535932,
          "CI95": [
            -0.00737504479131159,
            0.012131360110869262
          ],
          "N": 155
        },
        "interaction": -0.06800334221687904,
        "SE": 0.034120014404994736,
        "pvalue": 0.046254536248180336,
        "Holm_pvalue": 0.832581652467246
      },
      {
        "stage": 25,
        "a": 3,
        "b": 2,
        "context": "immediate_above_average_remaining_step",
        "tested": true,
        "high": {
          "mean": -0.014956024759731268,
          "MC_SE": 0.01789463515975907,
          "CI95": [
            -0.050029509672859046,
            0.020117460153396508
          ],
          "N": 68
        },
        "low": {
          "mean": -0.027885742124147208,
          "MC_SE": 0.017619991725573544,
          "CI95": [
            -0.06242092590627135,
            0.006649441657976939
          ],
          "N": 188
        },
        "interaction": 0.01292971736441594,
        "SE": 0.02511338439777016,
        "pvalue": 0.606655299034318,
        "Holm_pvalue": 1.0
      }
    ],
    "minimum_cohort": 16,
    "predeclared_family": 36,
    "interpretation": "Single-decision swaps share the complete prefix and preserve two distinct expert functions; downstream policy continues normally. Effects include nonlinear fusion interactions, not additive expert EIG."
  }
}
Useful E1–E4 specialization requires scientific inspection/replication; prediction accuracy or unequal routing alone is insufficient.
