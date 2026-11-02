# Matched single predictive control

One control satisfies bothA/B:1head,96/96tanh,12Gaussian components,13860parameters versus4heads+router13800parameters (0.435%difference). Same saved simulator/posterior data and same14000update schedule, including the phase-boundary Adam reset. It replaces only the predictive model; Bayesian inference/candidates/lookahead/horizon/officialbank stay unchanged.

Original heldout development comparison:
{
  "pimoe": {
    "NLL": 1.0168709895360302,
    "reference_NLL": 0.9944014829508,
    "NLL_excess": 0.02246950658523018,
    "EIG_RMSE": 0.08793658607755815,
    "EIG_bias": 0.027973055735617633,
    "KS": 0.034146581442451474,
    "coverage90": 0.9082845052083334,
    "selected_action_reference_EIG": 0.6472780444958482,
    "selected_action_regret": 0.004489071348759986,
    "ranking_pairs": 219,
    "ranking_accuracy": 0.9680365296803652,
    "wall_seconds": 0.15587419224902987
  },
  "single": {
    "NLL": 1.0179826162988237,
    "reference_NLL": 0.9944014829508,
    "NLL_excess": 0.023581133348023763,
    "EIG_RMSE": 0.07222303272424364,
    "EIG_bias": 0.023970959617014707,
    "KS": 0.039533845172410036,
    "coverage90": 0.905029296875,
    "selected_action_reference_EIG": 0.6303432851262878,
    "selected_action_regret": 0.021423830718320477,
    "ranking_pairs": 219,
    "ranking_accuracy": 0.9497716894977168,
    "wall_seconds": 0.21228873496875167
  }
}

Independent on-policy diagnostic comparison:
[
  {
    "model": "moe",
    "NLL": 0.8661086415604576,
    "KS": 0.020171198253574076,
    "coverage90": 0.9038550967261905,
    "EIG_RMSE": 0.1742735068064596,
    "EIG_bias": 0.049358346654472814,
    "selected_reference_EIG": 0.319138137173968,
    "selected_regret": 0.06924154629376213,
    "ranking_accuracy": 0.9345238095238095,
    "ranking_pairs": 840
  },
  {
    "model": "single",
    "NLL": 0.8671066295738221,
    "KS": 0.030068477367784043,
    "coverage90": 0.9015706380208331,
    "EIG_RMSE": 0.07095073597141408,
    "EIG_bias": 0.019453570325969193,
    "selected_reference_EIG": 0.3743099823698808,
    "selected_regret": 0.014069701097849382,
    "ranking_accuracy": 0.944047619047619,
    "ranking_pairs": 840
  }
]

Cost:
{
  "model": "matched_parameter_and_density_single_control",
  "training_seed": 101,
  "parameters": 13860,
  "experts": 1,
  "components": 12,
  "width": 96,
  "pilot_updates": 2000,
  "followon_updates": 12000,
  "total_updates": 14000,
  "phase_optimizer_reset_matches_original": true,
  "training_data_sha256": "f15a9464f894282921b2075f3c2d69b8ba0e46f660b1e4b08042bd8d81f921aa",
  "same_training_data_and_sample_order": true,
  "new_training_wall_seconds": 33.7138366503641,
  "original_pilot_wall_seconds": 5.2221463792957366,
  "conservative_complete_credit_seconds": 1244.345747041516,
  "credit_note": "Conservatively attribute entire original1210.63s development/data/pilots/follow-on credit plus new control training; not zero-cost shared data. Original phase-specific data time is unmeasured.",
  "shared_physical_data_work": {
    "physical_forward_evaluations_through_serialized_training_and_heldout_generation": 93890830,
    "simulator_count_scope": "Shared dataset preparation, not repeated for control; posterior/reference auxiliary phases separate"
  },
  "new_training_simulator_evaluations": 0,
  "checkpoint_sha256": "53eedc9b968b3003458463119bc70858c880b840ecb8d440764bed79593c445d",
  "hardware": "NVIDIA GeForce RTX 4090",
  "GPU_active_seconds": null,
  "peak_GPU_memory_bytes": 20857344,
  "selection": "predeclared last update; no official EIG selection",
  "source_commit": "75f74ed29e9f87cfb95bc80e6cd28d36f8573c25"
}

Official single evaluation: PENDING. No final paired improvement claim.

Equal nominal operations are not equal RTX4090/5080walltime. Measured hardware-specific times are kept separate; old reference dataset time is conservatively credited, never reported as free.
