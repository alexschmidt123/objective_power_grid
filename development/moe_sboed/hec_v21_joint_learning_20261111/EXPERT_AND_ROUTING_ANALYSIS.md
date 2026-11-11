# Expert and routing evidence

```json
{
  "expert_removal": {
    "1": {
      "n": 256,
      "mean": -0.030303154163892787,
      "MCSE": 0.0792991115747654,
      "CI95": [
        -0.18646773276778286,
        0.1258614244399973
      ],
      "p_two_sided": 0.7026777485793042,
      "Holm4_p": 0.7026777485793042
    },
    "2": {
      "n": 256,
      "mean": 0.12823986903929668,
      "MCSE": 0.10612221376278125,
      "CI95": [
        -0.08074772821961632,
        0.33722746629820965
      ],
      "p_two_sided": 0.2280063744548839,
      "Holm4_p": 0.4560127489097678
    },
    "3": {
      "n": 256,
      "mean": 0.23404469089265692,
      "MCSE": 0.08213581516684708,
      "CI95": [
        0.07229376192134929,
        0.39579561986396455
      ],
      "p_two_sided": 0.004736612392269886,
      "Holm4_p": 0.014209837176809658
    },
    "4": {
      "n": 256,
      "mean": 0.26316626368680124,
      "MCSE": 0.08621981219674066,
      "CI95": [
        0.09337267619728867,
        0.4329598511763138
      ],
      "p_two_sided": 0.0025113038423588423,
      "Holm4_p": 0.01004521536943537
    }
  },
  "new_correction_frozen_effect": {
    "n": 256,
    "mean": -0.050982445032276356,
    "MCSE": 0.01915428155015208,
    "CI95": [
      -0.08870317414687129,
      -0.013261715917681417
    ],
    "p_two_sided": 0.008269213683877358
  },
  "within_stage_gate_SD": [
    0.04217355698347092,
    0.08114165812730789,
    0.054877351969480515,
    0.07613395154476166
  ],
  "gate_means": [
    0.11439160257577896,
    0.3574613332748413,
    0.12122935801744461,
    0.4069150388240814
  ],
  "competency": {
    "designated_advantages": {
      "expert1": {
        "n": 256,
        "mean": 0.020302920277212283,
        "MCSE": 0.01432617080813991,
        "CI95": [
          -0.00790975932073237,
          0.04851559987515694
        ],
        "p_two_sided": 0.15764771747470888,
        "Holm4_p": 0.15764771747470888
      },
      "expert2": {
        "n": 256,
        "mean": 0.045531449827478344,
        "MCSE": 0.01085144107302053,
        "CI95": [
          0.024161592224275944,
          0.06690130743068075
        ],
        "p_two_sided": 3.7553930648194214e-05,
        "Holm4_p": 0.00011266179194458265
      },
      "expert3": {
        "n": 256,
        "mean": 0.24335957761286783,
        "MCSE": 0.015689643263194036,
        "CI95": [
          0.21246179729748593,
          0.27425735792824973
        ],
        "p_two_sided": 1.1470108355276772e-38,
        "Holm4_p": 4.588043342110709e-38
      },
      "expert4": {
        "n": 256,
        "mean": 0.03429027342449796,
        "MCSE": 0.015793711905393852,
        "CI95": [
          0.0031875496320420346,
          0.06539299721695388
        ],
        "p_two_sided": 0.03084353001093715,
        "Holm4_p": 0.0616870600218743
      }
    },
    "live_vs_original": {
      "expert1": {
        "n": 256,
        "mean": 0.007492506037428078,
        "MCSE": 0.0014249659438808349,
        "CI95": [
          0.004686305542467446,
          0.01029870653238871
        ],
        "p_two_sided": 3.08245690925776e-07,
        "Holm4_p": 1.232982763703104e-06
      },
      "expert2": {
        "n": 256,
        "mean": -0.003365380415095399,
        "MCSE": 0.0012556681353058225,
        "CI95": [
          -0.005838180946176838,
          -0.0008925798840139594
        ],
        "p_two_sided": 0.007838292526122064,
        "Holm4_p": 0.023514877578366192
      },
      "expert3": {
        "n": 256,
        "mean": 0.002030568747159935,
        "MCSE": 0.001590151207967054,
        "CI95": [
          -0.0011009328343491317,
          0.005162070328669001
        ],
        "p_two_sided": 0.20277586613570167,
        "Holm4_p": 0.20277586613570167
      },
      "expert4": {
        "n": 256,
        "mean": 0.0020794744602653506,
        "MCSE": 0.0009289283183653132,
        "CI95": [
          0.0002501261042756695,
          0.003908822816255032
        ],
        "p_two_sided": 0.026046899719339177,
        "Holm4_p": 0.052093799438678354
      }
    },
    "new_readouts_vs_constant": {
      "expert2": {
        "n": 256,
        "mean": -1.986255388794234,
        "MCSE": 0.0557788944194913,
        "CI95": [
          -2.0961013551490786,
          -1.8764094224393897
        ],
        "p_two_sided": 5.972406828328274e-101,
        "Holm2_p": 1.1944813656656548e-100
      },
      "expert4": {
        "n": 256,
        "mean": -0.6129398895136546,
        "MCSE": 0.030576682391239295,
        "CI95": [
          -0.6731548733376648,
          -0.5527249056896445
        ],
        "p_two_sided": 2.579344455879479e-54,
        "Holm2_p": 2.579344455879479e-54
      }
    },
    "predictive_teacher_observations": 36864,
    "analytic_prefix_excluded_from_learned_probes": true,
    "ridge_calibration_only": true,
    "dimension": 116,
    "legacy_reference_zero_padding_caveat": true,
    "targets": "E2heldoutrotatedhypothesisKL;E4secondstepinformationandtailvalue underapproximatebelief;notexactconditionalEIG"
  },
  "frozen_ablation_not_training_causality": true
}
```

Conditional resimulated native-return diagnostics:

```json
{
  "status": "COMPLETE",
  "cases": 128,
  "replicates": 2,
  "stages": [
    3,
    12,
    24
  ],
  "simulator_observations": 66144,
  "estimator": "pairedexpectedremainingnormalizednativevariationalreturn difference;prefixtermcancels;NOTexactconditionalMI",
  "conditioning": "theta retainedfromsamejointgeneratedprefix;noindependentpriorrestart",
  "groups": "stage-specificcalibrationmedianofapproximatebeliefvariancetrace",
  "calibration_medians": [
    2.902456521987915,
    1.6677294969558716,
    1.254337191581726
  ],
  "tests": {
    "stage3_signal0_expert1": {
      "status": "ESTIMATED",
      "distinct_cases": 67,
      "mean": 0.10657487053479721,
      "MCSE": 0.11191931462473842,
      "CI95": [
        -0.11687925083848462,
        0.33002899190807905
      ],
      "p_two_sided": 0.34444599077857196,
      "Holm24_p": 1.0
    },
    "stage3_signal0_expert2": {
      "status": "ESTIMATED",
      "distinct_cases": 67,
      "mean": 0.06129146000342583,
      "MCSE": 0.08905872257211367,
      "CI95": [
        -0.11652001668140141,
        0.23910293668825308
      ],
      "p_two_sided": 0.4937299196072652,
      "Holm24_p": 1.0
    },
    "stage3_signal0_expert3": {
      "status": "ESTIMATED",
      "distinct_cases": 67,
      "mean": -0.07395838645856771,
      "MCSE": 0.09800487502259465,
      "CI95": [
        -0.2696314328125483,
        0.1217146598954129
      ],
      "p_two_sided": 0.4531494395372681,
      "Holm24_p": 1.0
    },
    "stage3_signal0_expert4": {
      "status": "ESTIMATED",
      "distinct_cases": 67,
      "mean": -0.05161464214324951,
      "MCSE": 0.08243645008049695,
      "CI95": [
        -0.21620432519870816,
        0.11297504091220914
      ],
      "p_two_sided": 0.5333994095081096,
      "Holm24_p": 1.0
    },
    "stage3_signal1_expert1": {
      "status": "ESTIMATED",
      "distinct_cases": 61,
      "mean": 0.08397561507146867,
      "MCSE": 0.16112460880021004,
      "CI95": [
        -0.23832158898449124,
        0.4062728191274286
      ],
      "p_two_sided": 0.6041567965277166,
      "Holm24_p": 1.0
    },
    "stage3_signal1_expert2": {
      "status": "ESTIMATED",
      "distinct_cases": 61,
      "mean": -0.199856556829859,
      "MCSE": 0.20932988508801958,
      "CI95": [
        -0.61857867005392,
        0.218865556394202
      ],
      "p_two_sided": 0.34353781060249516,
      "Holm24_p": 1.0
    },
    "stage3_signal1_expert3": {
      "status": "ESTIMATED",
      "distinct_cases": 61,
      "mean": -0.14992055443466687,
      "MCSE": 0.20741902232898532,
      "CI95": [
        -0.5648203730436635,
        0.2649792641743297
      ],
      "p_two_sided": 0.472616557190223,
      "Holm24_p": 1.0
    },
    "stage3_signal1_expert4": {
      "status": "ESTIMATED",
      "distinct_cases": 61,
      "mean": -0.08806364311546576,
      "MCSE": 0.08937680893504997,
      "CI95": [
        -0.2668438793668309,
        0.0907165931358994
      ],
      "p_two_sided": 0.3284297621292468,
      "Holm24_p": 1.0
    },
    "stage12_signal0_expert1": {
      "status": "ESTIMATED",
      "distinct_cases": 72,
      "mean": -0.03542775091611677,
      "MCSE": 0.06491088071536905,
      "CI95": [
        -0.1648563710195454,
        0.09400086918731185
      ],
      "p_two_sided": 0.5869206627190666,
      "Holm24_p": 1.0
    },
    "stage12_signal0_expert2": {
      "status": "ESTIMATED",
      "distinct_cases": 72,
      "mean": 0.1617872610481249,
      "MCSE": 0.10486656749034577,
      "CI95": [
        -0.04731073570798572,
        0.37088525780423554
      ],
      "p_two_sided": 0.12732696546211247,
      "Holm24_p": 1.0
    },
    "stage12_signal0_expert3": {
      "status": "ESTIMATED",
      "distinct_cases": 72,
      "mean": -0.015020098951127794,
      "MCSE": 0.06361574481210597,
      "CI95": [
        -0.14186629140978624,
        0.11182609350753064
      ],
      "p_two_sided": 0.8140297203993038,
      "Holm24_p": 1.0
    },
    "stage12_signal0_expert4": {
      "status": "ESTIMATED",
      "distinct_cases": 72,
      "mean": 0.08895828988817003,
      "MCSE": 0.0680404418424592,
      "CI95": [
        -0.04671049786888751,
        0.22462707764522757
      ],
      "p_two_sided": 0.19528393448531695,
      "Holm24_p": 1.0
    },
    "stage12_signal1_expert1": {
      "status": "ESTIMATED",
      "distinct_cases": 56,
      "mean": 0.14397086389362812,
      "MCSE": 0.09123260110818107,
      "CI95": [
        -0.038863354423121665,
        0.3268050822103779
      ],
      "p_two_sided": 0.12028579546694088,
      "Holm24_p": 1.0
    },
    "stage12_signal1_expert2": {
      "status": "ESTIMATED",
      "distinct_cases": 56,
      "mean": -0.020403960985796794,
      "MCSE": 0.11999065067159208,
      "CI95": [
        -0.2608705985076711,
        0.2200626765360775
      ],
      "p_two_sided": 0.8655977160968932,
      "Holm24_p": 1.0
    },
    "stage12_signal1_expert3": {
      "status": "ESTIMATED",
      "distinct_cases": 56,
      "mean": 0.039834407291242054,
      "MCSE": 0.09929787173992292,
      "CI95": [
        -0.15916297456086514,
        0.23883178914334924
      ],
      "p_two_sided": 0.6898561379456932,
      "Holm24_p": 1.0
    },
    "stage12_signal1_expert4": {
      "status": "ESTIMATED",
      "distinct_cases": 56,
      "mean": 0.02624484178211008,
      "MCSE": 0.10069712504504295,
      "CI95": [
        -0.175556706356623,
        0.22804638992084317
      ],
      "p_two_sided": 0.7953500663561259,
      "Holm24_p": 1.0
    },
    "stage24_signal0_expert1": {
      "status": "ESTIMATED",
      "distinct_cases": 76,
      "mean": -0.002786060305018174,
      "MCSE": 0.01620465450952868,
      "CI95": [
        -0.03506738745831239,
        0.029495266848276042
      ],
      "p_two_sided": 0.8639559240227837,
      "Holm24_p": 1.0
    },
    "stage24_signal0_expert2": {
      "status": "ESTIMATED",
      "distinct_cases": 76,
      "mean": -0.06952777308853049,
      "MCSE": 0.07885800601004835,
      "CI95": [
        -0.22662097672146952,
        0.08756543054440853
      ],
      "p_two_sided": 0.3807651490429156,
      "Holm24_p": 1.0
    },
    "stage24_signal0_expert3": {
      "status": "ESTIMATED",
      "distinct_cases": 76,
      "mean": -0.035246273404673525,
      "MCSE": 0.10541667610335666,
      "CI95": [
        -0.24524706093792698,
        0.1747545141285799
      ],
      "p_two_sided": 0.739047164574798,
      "Holm24_p": 1.0
    },
    "stage24_signal0_expert4": {
      "status": "ESTIMATED",
      "distinct_cases": 76,
      "mean": 0.038205741267455254,
      "MCSE": 0.07927366554273296,
      "CI95": [
        -0.11971549861587637,
        0.19612698115078686
      ],
      "p_two_sided": 0.6312467210266972,
      "Holm24_p": 1.0
    },
    "stage24_signal1_expert1": {
      "status": "ESTIMATED",
      "distinct_cases": 52,
      "mean": 0.04058498659959206,
      "MCSE": 0.04278360448699198,
      "CI95": [
        -0.045306683404104806,
        0.12647665660328894
      ],
      "p_two_sided": 0.3472921873835401,
      "Holm24_p": 1.0
    },
    "stage24_signal1_expert2": {
      "status": "ESTIMATED",
      "distinct_cases": 52,
      "mean": -0.037244400152793296,
      "MCSE": 0.09347264623391095,
      "CI95": [
        -0.22489856770046657,
        0.15040976739488
      ],
      "p_two_sided": 0.6919600359337349,
      "Holm24_p": 1.0
    },
    "stage24_signal1_expert3": {
      "status": "ESTIMATED",
      "distinct_cases": 52,
      "mean": 0.12170984309453231,
      "MCSE": 0.08507037636928344,
      "CI95": [
        -0.04907606383910093,
        0.2924957500281655
      ],
      "p_two_sided": 0.1586163098999679,
      "Holm24_p": 1.0
    },
    "stage24_signal1_expert4": {
      "status": "ESTIMATED",
      "distinct_cases": 52,
      "mean": 0.09554887391053714,
      "MCSE": 0.07573381009530254,
      "CI95": [
        -0.05649309410097385,
        0.2475908419220481
      ],
      "p_two_sided": 0.2128180214271892,
      "Holm24_p": 1.0
    }
  },
  "canonical_identical_continuation_bitwise_consistency": true,
  "changed_policy_observations_resimulated": true
}
```

Frozen weights,output diversity or gate variation are not evidence of useful competence. Learned-representation tasks exclude exact analytic semantic bypasses;calibration chooses regressors/rivals without heldout outcomes. Retention contrasts are not equivalence proofs. Frozen anchors stay byte-identical,buttheir eventual usefulness is measured separately. Conditional donor effects estimate grouped expected native return,not exact per-history EIG. Distribution shift can affect single/removal/correction-off interventions. No causal training interaction claim from frozen ablations alone.
