# Bounded development pilot

No improvement is established. Pilot weights discarded.

{
  "PASS": true,
  "foundation": "13681d7091189a4e68056e6a2864dc58f9d5700b",
  "updates_per_policy": 100,
  "paired_training_randomness": true,
  "main_weights_reused": false,
  "specialization_adjustments": 30,
  "adjusted_stages": [
    1,
    10,
    20,
    29
  ],
  "CGS_minus_matched_continued_v1": {
    "difference": -0.12925260309092548,
    "paired_MC_SE": 0.08141939864050954,
    "CI95": [
      -0.2888346244263242,
      0.030329418244473216
    ],
    "N": 512
  },
  "CGS_minus_frozen_v1": {
    "difference": -0.15399538980723929,
    "paired_MC_SE": 0.08415666126652108,
    "CI95": [
      -0.3189424458896206,
      0.010951666275142019
    ],
    "N": 512
  },
  "results": {
    "frozen_v1": {
      "EIG": 10.838089590198386,
      "MC_SE": 0.07265574216451552,
      "metadata": {
        "seed": 97665003,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "06290a00570ef68a4486dd3652d1d441bf79d6afe05bf427761831afec909a88",
        "seconds": 3.319941744906828,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    },
    "continued_v1": {
      "EIG": 10.813346803482073,
      "MC_SE": 0.06848490588154521,
      "metadata": {
        "seed": 97665003,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "06290a00570ef68a4486dd3652d1d441bf79d6afe05bf427761831afec909a88",
        "seconds": 3.319432646036148,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    },
    "specialized_v1": {
      "EIG": 10.684094200391147,
      "MC_SE": 0.0742719010785548,
      "metadata": {
        "seed": 97665003,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "06290a00570ef68a4486dd3652d1d441bf79d6afe05bf427761831afec909a88",
        "seconds": 3.3229209140408784,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    }
  },
  "wall_seconds": 144.70005294599105,
  "parameter_stage_propagations": [
    12006000000,
    12006000000
  ],
  "specialization_extra_seconds": 5.110002521192655,
  "selection": "Predeclared single procedure; no hyperparameter selection; pilot weights discarded"
}
