# Proper role-specific learning

Main loss is -mean(terminal full-horizon sequential sPCE). All30 measurements and complete simulator/history gradients contribute. Each expert has its own realized information-density target and proper squared prediction loss. Targets and the entire shared history embedding supplied to auxiliary predictors are stop-gradient. Thus role loss e updates ONLY expert e's private feature parameters and prediction head; encoder/router/design head and other experts receive no role-e auxiliary gradient. The main sPCE gradient still updates all selected experts, router weights, shared encoder and joint action head through the complete sequential simulator path.

Prediction targets condition on observed history under the current on-policy action and continuation policy. They may use hidden parameters and future observations offline ONLY. Targets move as the policy learns; this is a conditional value-prediction mechanism, not a proof that an expert causally chooses better experiments. Future prediction outputs have an analytic zero boundary at the final stage; expert routing itself is not manually assigned to early/late stages.

Per-role mean/RMS normalization is frozen from a pretraining-only diagnostic (scale floor.1nats); no realized target is clipped. Alpha is min(1,.1*primary-expert-gradient-norm/auxiliary-expert-gradient-norm), measured once before any held-out performance evaluation. This is an explicitly heuristic gradient-scale choice, not an MI identity or a guarantee of long-run gradient balance. The no-role-loss control tests whether it helps EIG. No router diversity/usage regularizer is imposed.

Measured calibration:

{
  "alpha": 0.8408488344989506,
  "primary_expert_gradient_norm": 0.03555245101585599,
  "raw_auxiliary_expert_gradient_norm": 0.004228162014048717,
  "weighted_aux_to_primary_gradient_ratio": 0.1,
  "role_mean": [
    0.0025500688817452627,
    0.08401197417567095,
    0.009186401460937984,
    0.41599305944333675
  ],
  "role_scale": [
    0.1,
    0.39348309875662835,
    0.1,
    0.7332501661142823
  ],
  "initial_target_correlation": [
    [
      1.0,
      0.0908503377704226,
      0.0360027072164827,
      0.044142296883270725
    ],
    [
      0.09085033777042262,
      0.9999999999999999,
      0.11697514201026804,
      0.04841366464985376
    ],
    [
      0.0360027072164827,
      0.11697514201026804,
      1.0,
      0.2684009958900003
    ],
    [
      0.044142296883270725,
      0.04841366464985376,
      0.2684009958900004,
      0.9999999999999999
    ]
  ],
  "training_diagnostic_seed": 86404021,
  "no_EIG_tuning": true
}


Held-out role prediction results:

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