# DAD audit

{
  "status": "PASS_NO_IMPLEMENTATION_DEFECT_FOUND_IN_TESTED_COMPONENTS",
  "checkpoint_hash_and_update1000_tensor_identity": "PASS",
  "truth_inclusive_terminal_joint_spce_and_negative_loss": "PASS",
  "end_to_end_T30_gradient_tests": [
    {
      "parameter": "encoder.0.weight",
      "autograd_loss_gradient": -1.5142409856977286,
      "finite_difference_loss_gradient": -1.5142409888202477,
      "error": 3.122519132148227e-09
    },
    {
      "parameter": "encoder.2.weight",
      "autograd_loss_gradient": -5.719711481482349,
      "finite_difference_loss_gradient": -5.719614625210311,
      "error": 9.68562720382593e-05
    },
    {
      "parameter": "emitter.bias",
      "autograd_loss_gradient": -3.1208584514656055,
      "finite_difference_loss_gradient": -3.120858426974848,
      "error": 2.449075742205764e-08
    }
  ],
  "architecture": "LocationDAD: raw[y,action] tokens3\u2192256ReLU\u219216 sum;16\u21922 emitter;first-designrev1",
  "architecture_parameters": 5170,
  "history_no_lookahead": "PASS",
  "contrast_distribution": "fresh independent prior particles, truth explicitly included in denominator",
  "physical": "iidstandardnormal4parameters,logsignal,.5Gaussiannoise,T30,[-4,4]^2",
  "optimizer": "Adam5e-5 betas(.8,.998), native gamma.98 every1000 after first update, no extra clipping",
  "evaluation_checkpoint_and_common_scorer_bank": "PASS",
  "old_selected_updates": 1000,
  "old_selected_primary": 2000000,
  "old_training_seconds": 239.00257333298214,
  "old_eig": 8.150681345146676,
  "old_mcse": 0.05069002991859373,
  "diagnostic_small_contrast_not_final": {
    "seed": 202611019181,
    "cases": 64,
    "L": 2000,
    "mean": 7.064319483070806,
    "upper_bound": 7.601402334583733,
    "action_boundary_fraction": 0.0
  },
  "assessment": "Only2% of50k native update schedule ran. Insufficient training is plausible; low EIG alone is not proof of incorrect implementation or a causal diagnosis. Original run lacked enough checkpoints to establish convergence. The new curve and paired evaluation will test training sufficiency.",
  "affected_previous_experiments": "No defect established; no retrospective invalidation or alteration"
}