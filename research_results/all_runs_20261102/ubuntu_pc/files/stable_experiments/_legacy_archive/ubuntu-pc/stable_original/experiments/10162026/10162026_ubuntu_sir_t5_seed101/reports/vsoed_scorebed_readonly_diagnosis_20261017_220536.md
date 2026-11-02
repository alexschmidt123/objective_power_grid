# SIR vsOED and ScoreBED saved-artifact diagnosis

Read-only analysis of saved SIR developmentseed101; no simulations, training, GPU jobs or source/config changes

```json
{
  "timestamp": "2026-10-17T22:05:36.396952-04:00",
  "scope": "Read-only analysis of saved SIR developmentseed101; no simulations, training, GPU jobs or source/config changes",
  "methods": {
    "vsoed": {
      "training_record": "/home/qiqi/Documents/objective_power_grid/experiments/10162026/10162026_ubuntu_sir_t5_seed101/runs/vsoed_seed101/models/vsoed_training.json",
      "training_record_sha256": "8fb82699dcf244960109e86bcbaf74960317338ed8e58fa54fc95867355072e3",
      "evaluation_path": "/home/qiqi/Documents/objective_power_grid/experiments/10162026/10162026_ubuntu_sir_t5_seed101/runs/vsoed_seed101/eval/vsoed_seed1001.json",
      "evaluation_sha256": "e88fc069c028c625d07560ceca5aa294063fcdd72d79dda6f074e1265af8ec80",
      "eig": 1.5522322818660954,
      "mean_measurement_times": [
        99.54144111633592,
        99.70000000000285,
        99.79999999999714,
        99.89999999999638,
        100.0
      ],
      "sd_measurement_times": [
        2.9132252166164108e-12,
        2.8421709430404007e-12,
        2.8563817977556027e-12,
        3.623767952376511e-12,
        0.0
      ],
      "invalid_spacing": 0,
      "validation": [
        {
          "episodes": 125120,
          "eig": 1.1802207302703625,
          "mc_standard_error": 0.18359993443847256
        },
        {
          "episodes": 250176,
          "eig": 1.1802347759022815,
          "mc_standard_error": 0.18359931715702096
        },
        {
          "episodes": 375232,
          "eig": 1.180240645217855,
          "mc_standard_error": 0.18359905912412341
        },
        {
          "episodes": 500288,
          "eig": 1.18024026032768,
          "mc_standard_error": 0.18359907604663026
        },
        {
          "episodes": 625344,
          "eig": 1.1802508937867844,
          "mc_standard_error": 0.18359860844391396
        },
        {
          "episodes": 750400,
          "eig": 1.1802819865654381,
          "mc_standard_error": 0.18359724019367862
        },
        {
          "episodes": 875456,
          "eig": 1.1802819865654381,
          "mc_standard_error": 0.18359724019367862
        },
        {
          "episodes": 1000000,
          "eig": 1.1803357856427996,
          "mc_standard_error": 0.18359486936693262
        }
      ],
      "first_logged_mean_design_above99": {
        "Epoch": "57",
        "Reward/MeanReward": "0.23007875680923462",
        "Reward/StdReward": "0.5221269726753235",
        "Design/MeanDesign": "[99.1063461303711]",
        "Design/StdDesign": "[1.7887170314788818]",
        "ReplayBuffer/buffer_size": "7616"
      },
      "confirmed": "Final policy selects an effectively identical terminal-time sequence across2048heldout trajectories; validation already poor at125120episodes and remains flat.",
      "unresolved": "Actor saturation/projection, history scaling, variational reward and train/eval time-conversion contributions are not yet causally separated.",
      "next_test": "Paired native actor/adapter forward and gradient checks through chronological feasible-time projection; no training or hyperparameter change yet."
    },
    "scorebed": {
      "training_record": "/home/qiqi/Documents/objective_power_grid/experiments/10162026/10162026_ubuntu_sir_t5_seed101/runs/scorebed_seed101/models/scorebed_training.json",
      "training_record_sha256": "913dda11a31b986b53807709427a29466ec31071b6ce14ec8ffaaa0ed571e83d",
      "evaluation_path": "/home/qiqi/Documents/objective_power_grid/experiments/10162026/10162026_ubuntu_sir_t5_seed101/runs/scorebed_seed101/eval/scorebed_seed1001.json",
      "evaluation_sha256": "e47ecf72d37f86a6833a9b72bfcbca417457355b1cea8c9a6531e4708b72be78",
      "eig": 4.229262700094608,
      "mean_measurement_times": [
        50.288322368782254,
        75.88522913603639,
        88.60513790613594,
        94.78454626582977,
        97.73015202359869
      ],
      "sd_measurement_times": [
        2.842170943040401e-14,
        0.01892923104818059,
        0.02342475902903472,
        0.01905954436018326,
        0.012846007940399999
      ],
      "invalid_spacing": 0,
      "validation": [
        {
          "episodes": 703068,
          "eig": 3.7335065086914088,
          "mc_standard_error": 0.2326786694499282
        },
        {
          "episodes": 828124,
          "eig": 3.7199624830931555,
          "mc_standard_error": 0.23334660122399917
        },
        {
          "episodes": 953180,
          "eig": 3.7065821236480465,
          "mc_standard_error": 0.2339903950338308
        },
        {
          "episodes": 1000000,
          "eig": 3.701612221690205,
          "mc_standard_error": 0.23421478348268468
        }
      ],
      "saved_configuration": "/home/qiqi/Documents/objective_power_grid/experiments/10162026/10162026_ubuntu_sir_t5_seed101/runs/scorebed_seed101/models/scorebed_official_config.json",
      "saved_configuration_sha256": "a8de4f66338c32184ba6b25411e3f71c7448edb9280ad117b2e40033d7ff08ac",
      "gradient_norms": {
        "score": {
          "logged_updates": 217,
          "quantiles_min_median_p95_max": [
            1.753246119697137e-10,
            7.654467051467333e-10,
            1.904370828673633e-09,
            2.595874058372273e-09
          ]
        },
        "policy": {
          "logged_updates": 92,
          "quantiles_min_median_p95_max": [
            1.0577881903728326e-09,
            1.1352254136731688e-09,
            1.1895612828105584e-09,
            1.23150700748198e-09
          ]
        }
      },
      "observation_dim": 1,
      "design_dim": 1,
      "default_potential_axis_width": 2,
      "potential_hidden_dim": null,
      "source_warning": "/home/qiqi/Documents/objective_power_grid/src/research/vendor/scorebed/policy_learning/score_nets.py:546",
      "source_sha256": "68281736150ed88c7856e1a584bf9c58d3018e52cb244163e5f8efec1bb5c4dc",
      "confirmed": "ScalarSIR uses2-channel LayerNorm potential with native widening option absent; native source documents near-constant-sign degeneration for this case. Recorded score/policy gradients are approximately1e-9 and actions remain nearly observation-independent.",
      "causal_status": "Structural problem identified, but improvement from enabling the native widening option has not yet been tested.",
      "next_test": "CPU/native gradient comparison at identical weights/data/seeds for the existing potential-head option, with no policy training; freeze one justified SIR-specific correction only after evidence.",
      "scope": "SIR-only scalar-design issue; locationfinding has3potentialchannels and this diagnosis does not apply to its active budget-allocation trial."
    }
  },
  "conclusion": "Two poor SIR trained policies; not evidence that the published algorithms generally fail. Preserve results and inspect adaptations before journal ranking.",
  "labpc_campaign_changed": false
}
```
