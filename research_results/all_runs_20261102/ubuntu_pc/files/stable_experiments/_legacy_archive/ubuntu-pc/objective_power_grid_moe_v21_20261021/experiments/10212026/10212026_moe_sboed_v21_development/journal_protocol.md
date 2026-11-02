{
  "model": "MoE-sBOED-v2.1",
  "seed": 101,
  "horizon": 30,
  "one_attempt": true,
  "no_sweep": true,
  "changes": [
    "router gamma denominator500k->3.2M residual episodes",
    "router Adam LR0.2 times unchanged expert LR"
  ],
  "unchanged": [
    "all4 independent64/64tanh experts",
    "full native frozen base and shared encoder",
    "alpha1, residual init std.001",
    "sPCE objective,L65536,chunk8192",
    "component clip5,Adam betas,LRdecay,regularization",
    "simulator,prior,likelihood,bounds"
  ],
  "reference_max_updates": 50000,
  "development": {
    "seed": 1004,
    "systems": 256,
    "contrasts": 65536,
    "selection": "latest budget-eligible; no best EIG selection"
  },
  "final_comparison": {
    "seed": 1001,
    "systems": 2000,
    "contrasts": 1000000,
    "evaluate_once_after_training": true,
    "previously_consumed_reference_bank": true
  },
  "budget": {
    "seconds": 14400,
    "development_elapsed_seconds": 1706.5388761290233,
    "conservative_launch_reserve_seconds": 120,
    "development_credit_seconds": 1826.5388761290233,
    "base_phase_seconds": 361.19603366503725,
    "base_prior_v2_development_not_recharged": true,
    "includes_validation": true
  },
  "bank_sha256": {
    "location_cases.npz": "cbd2632af41a8c6ca8cbe55066d77b98791565b7ac301b0bea1965e84ec50f5b",
    "location_contrasts.npy": "703671b0fb01ddf74dfe60d51e44914c71647c92e7e5ee51d111a9acfcecfde8"
  },
  "training_attempts_permitted": 1,
  "source_base_commit": "75d2bf2aad54f77965fb79c9d04bee7d9e12dafe",
  "promotion": "Only after correctness and improved development performance; no overwrite or merge of active LabPC source"
}