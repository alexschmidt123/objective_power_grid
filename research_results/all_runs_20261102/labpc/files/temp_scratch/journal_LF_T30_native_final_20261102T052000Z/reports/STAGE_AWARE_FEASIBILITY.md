# Production feasibility

ScoreBED full-shape probe: {
  "status": "PASS",
  "production_batch": 1024,
  "horizon": 30,
  "restarts": 5,
  "native_score_scheduler": 50000,
  "native_policy_scheduler_per_restart": 20000,
  "score_unit_seconds_upper": 0.08192484697792679,
  "policy_unit_seconds_upper": 0.023206728976219893,
  "projected_seconds_including_50pct_margin_and600s": 10225.372869777493,
  "measured_probe_all_phases_seconds": 152.40901871595997,
  "five_restart_initialization_and_selection": true,
  "selected_restart_export_bitwise_match": true,
  "profile_updates_only_not_formal_training": true,
  "source": "score_profile.py: counts shortened only for bounded readiness; production optimizer/network/sampler/T30/B and scheduler horizons retained"
}
DAD: archived native50k completed11959s with same source/config on4090. vsOED G-I: archived10001cycles completed6807s, unchanged native profile. RL2M: completed27451s, unchanged target/native settings; no convergence claim. Fixed B2000/L2000 receives an exact-production4-update profile before launch.
All jobs have10.5h soft native-unit cutoff and<12h parent wall watchdog. Imports, init, training, checkpoints and native selection are included in parent wall. Evaluation costs separately preserved.
