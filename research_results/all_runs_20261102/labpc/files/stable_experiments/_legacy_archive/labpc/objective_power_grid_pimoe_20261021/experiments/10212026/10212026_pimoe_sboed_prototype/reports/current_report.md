# PIMoE-sBOED seed101 prototype

FINAL_EVALUATION_RUNNING_OR_PENDING

Core implemented;8 focused tests passed. Dedicated isolated LabPC branch, active main untouched.

Frozen charged development/training: 1210.632s /14400s. Fourteen thousand MoE optimizer updates (2000pilot+12000fixed follow-on), plus2000internal single-control updates, all charged.

## Matched pilot comparison

{
  "predictive_moe": {
    "NLL": 1.0007790730764783,
    "reference_NLL": 0.9944014829508,
    "NLL_excess": 0.006377590125678244,
    "EIG_RMSE": 0.06612940818122101,
    "EIG_bias": 0.02192005135156852,
    "KS": 0.03200530490755035,
    "coverage90": 0.9037272135416666,
    "selected_action_reference_EIG": 0.6517671158446083,
    "selected_action_regret": 0.0,
    "ranking_pairs": 219,
    "ranking_accuracy": 1.0,
    "wall_seconds": 0.18862642301246524
  },
  "single": {
    "NLL": 0.9999757507604018,
    "reference_NLL": 0.9944014829508,
    "NLL_excess": 0.0055742678096019965,
    "EIG_RMSE": 0.05763753121689897,
    "EIG_bias": 0.01749156758160723,
    "KS": 0.03029970622717228,
    "coverage90": 0.9041341145833334,
    "selected_action_reference_EIG": 0.6517671158446083,
    "selected_action_regret": 0.0,
    "ranking_pairs": 219,
    "ranking_accuracy": 0.9954337899543378,
    "wall_seconds": 0.18672897620126605
  },
  "capacity": {
    "moe_parameters": 13800,
    "single_parameters": 13860,
    "single_width": 96,
    "relative_difference": 0.004347826086956522,
    "same_total_gaussian_components": 12,
    "same_training_data_and_updates": true
  }
}

The MoE did not outperform the parameter-matched single model in predictive NLL or EIG RMSE. Its confident ranking accuracy is slightly higher, with identical selected-action reference EIG on this small diagnostic set. No statistically established architectural value.

## Fixed follow-on model

{
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
}

The extra fixed training degraded held-out NLL and EIG RMSE relative to the pilot. No best-development checkpoint substitution was made; final uses the predeclared last checkpoint.

## Inference limitations

Independent2048/4096particle/rejuvenation checks passed on four held-out states from one episode. This does not validate all high-information on-policy posteriors or rule out missed modes. The complete official trajectory evaluation is required. Two-step lookahead has no demonstrated general benefit from the two-state internal check.

## Accounting limitation

{
  "charged_development_training_seconds": 1210.631910391152,
  "training_allowance_seconds": 14400,
  "source_commit": "75f74ed29e9f87cfb95bc80e6cd28d36f8573c25",
  "optimizer_updates": {
    "pimoe_pilot_training": 2000,
    "pimoe_final_training": 12000,
    "single_control_training": 2000
  },
  "fresh_complete_training_histories": 6,
  "fresh_complete_heldout_diagnostic_histories": 3,
  "conditional_reference": "Held-out posterior-state diagnostics are charged to development",
  "warning": "resource_ledger.json from launcher spans end-to-end lifecycle if evaluation finishes; training charge is frozen here before evaluation. Do not interpret total end-to-end simulator counter as training-only.",
  "GPU_active_seconds": null,
  "training_only_forward_counter": null,
  "training_only_likelihood_counter": null,
  "missing_counter_note": "Training/evaluation physical-work split was not snapshotted at phase boundary; unknown, not zero. Runtime budget boundary is measured exactly."
}