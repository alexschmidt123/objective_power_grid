# Read-only correctness audit

2026-10-17T22:00:58.967959-04:00 through 2026-10-17T22:01:00.927439-04:00

Campaign: `/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction`

Overall: **FAIL_REQUIRES_REVIEW**. No training or queue change; no GPU jobs, simulations, or code/configuration edits. Seed101 finished before audit; resmoe seed103 active at report completion.

| Check | Result |
|---|---|
| 1. ResMoE101 ST treatment and original frozen500k base | PASS |
| 2. All15 MoE jobs: intended routing and architecture | PASS |
| 3. 500k+500k residual accounting and exact shared bases | FAIL |
| 4. Five existing STMoE runs reused, not retrained | PASS |
| 5. Identical1M ScoreBED allocation and native objectives | PASS |
| 6. Development-only evaluation; no old-final selection | PASS |
| 7. Frozen hashes, command/seed mapping and checkpoint rule | PASS |
| 8. Historical ScoreBED101 slow evaluation explained | PASS |
| 9. Failure/partial completion gates and no historical overwrite | FAIL |

## 1. ResMoE101 ST treatment and original frozen500k base — PASS

```json
{
  "note": "Seed101 completed before this audit; seed102 is now active. Final checkpoint proves full frozen-base preservation.",
  "run": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed101/resmoe",
  "training_record": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed101/resmoe/models/resmoe_training.json",
  "effective_settings": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed101/resmoe/models/resmoe_effective_settings.json",
  "checkpoint_sha256": "e6d34ac24339727ad26a2eda5d78e88737aa68552b40ca9ddf68ba6b656bfa9e",
  "base_checkpoint": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_location_residual_1m_total_v2/development_seed101/base/models/sum_single.pt",
  "base_sha256": "9e8793ab5ffa27e6debd7414ee2135d9b9a3ee0c219a67d08fa3da48367c820e",
  "base_tensors_bitwise_equal": 11,
  "trace_samples": [
    {
      "update": 1,
      "episodes": 64,
      "mode": "straight_through",
      "base_gradient": 0.0
    },
    {
      "update": 2,
      "episodes": 128,
      "mode": "straight_through",
      "base_gradient": 0.0
    },
    {
      "update": 7813,
      "episodes": 500000,
      "mode": "straight_through",
      "base_gradient": 0.0
    }
  ]
}
```

## 2. All15 MoE jobs: intended routing and architecture — PASS

```json
{
  "configurations": [
    "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/configs/location_500000.yaml",
    "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/configs/location_1000000.yaml"
  ],
  "verified_jobs": 15,
  "architecture_sources": [
    "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/frozen_source/src/research/moe_family.py",
    "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/frozen_source/src/research/strategy_moe.py"
  ],
  "forced_reroute_and_gradient_test_record": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/metadata/correctness.json",
  "public_tests": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/logs/public_compatibility.log",
  "note": "Independent64/64tanh experts retained; shared residual base kept; learned gate remains present/trainable. All four proposals computed for ST backward; hard inference dispatch remains one expert."
}
```

## 3. 500k+500k residual accounting and exact shared bases — FAIL

```json
"AssertionError: "
```

## 4. Five existing STMoE runs reused, not retrained — PASS

```json
{
  "reuse_registry": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/metadata/reused_ST_moe.json",
  "five_original_checkpoints_verified": [
    {
      "seed": 101,
      "checkpoint": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_st_replication_score_reverse_cross/runs/moe_seed101/models/moe.pt",
      "sha256": "7f7635777ca966ba61a2795ebffab099112a28170335e3903145972f11924bd2"
    },
    {
      "seed": 102,
      "checkpoint": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_st_replication_score_reverse_cross/runs/moe_seed102/models/moe.pt",
      "sha256": "6afcdbff999aaa068de3e51bd488ee715982543cdfbf08c60758e7199c20c71c"
    },
    {
      "seed": 103,
      "checkpoint": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_moe_st_routing_dev_103_105/source/experiments/10172026/10172026_moe_st_dev/seed103/models/moe.pt",
      "sha256": "c502949b149377ccf6e965a88bbc89458340f1ed153917d27f4404df852d6064"
    },
    {
      "seed": 104,
      "checkpoint": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_st_replication_score_reverse_cross/runs/moe_seed104/models/moe.pt",
      "sha256": "bdd52dfde5604c93884f2c69448e894f11f2e2f58dbc5b3569c584a9a0c04d0a"
    },
    {
      "seed": 105,
      "checkpoint": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_moe_st_routing_dev_103_105/source/experiments/10172026/10172026_moe_st_dev/seed105/models/moe.pt",
      "sha256": "ade7e036368847abbf3f4728d841bec2af172e6ed336796673251c6a97eac780"
    }
  ],
  "moe_training_queue_slots": 0,
  "prior_action_replay_test": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/metadata/correctness.json"
}
```

## 5. Identical1M ScoreBED allocation and native objectives — PASS

```json
{
  "allocation_file": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/protocol_scorebed_allocation.json",
  "allocation": [
    10014,
    792000,
    197986
  ],
  "sum": 1000000,
  "five_seeds": [
    101,
    102,
    103,
    104,
    105
  ],
  "single_configuration": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/configs/location_1000000.yaml",
  "verified_pinned_vendor_files": 79,
  "native_worker": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/frozen_source/src/research/scorebed_worker.py",
  "preserved_objectives": "Native conditional-score matching + estimate_type score policy gradient; no alternative EIG or MSE loss.",
  "smoke_record": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/metadata/scorebed_smoke.json",
  "normalization_saved": "scorebed_normalization.npz plus native score variables/config/initial-state hashes",
  "policy_updates": 3094,
  "note": "Allocation hypothesis only, not guaranteed fix or convergence."
}
```

## 6. Development-only evaluation; no old-final selection — PASS

```json
{
  "executed_command_records": [
    "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed103/resmoe_command.json",
    "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed101/resmoe_command.json",
    "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed102/resmoe_command.json"
  ],
  "controller": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/frozen_source/tools/journal_campaign.py",
  "development_seed": 1001,
  "diagnostic_validation_seed": 771001,
  "old_final_seed8831001": "Historical provenance/previously-used list only; no corrective evaluation artifacts or job command",
  "new_final_seed": null,
  "new_bank": "Not generated; requires later method/analysis freeze"
}
```

## 7. Frozen hashes, command/seed mapping and checkpoint rule — PASS

```json
{
  "scientific_and_script_hashes_verified": 325,
  "native_nonPython_config_hashes_verified": 45,
  "freeze": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location_freeze.json",
  "manifest_sha256": "2f931b9e1b60323f8146ac4bff73451f9e260a33994d2762eee46ae026bf20a8",
  "executed_commands": [
    {
      "method": "resmoe",
      "seed": 103,
      "path": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed103/resmoe_command.json",
      "command": [
        "bash",
        "run.sh",
        "--config",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/configs/location_500000.yaml",
        "--journal",
        "--seed",
        "103",
        "--eval-seeds",
        "1001",
        "--eig-only",
        "--source-check-manifest",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/source_manifest.json",
        "--output",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed103/resmoe",
        "--variant",
        "resmoe",
        "--base-checkpoint",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed103/base/models/sum_single.pt"
      ]
    },
    {
      "method": "resmoe",
      "seed": 101,
      "path": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed101/resmoe_command.json",
      "command": [
        "bash",
        "run.sh",
        "--config",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/configs/location_500000.yaml",
        "--journal",
        "--seed",
        "101",
        "--eval-seeds",
        "1001",
        "--eig-only",
        "--source-check-manifest",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/source_manifest.json",
        "--output",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed101/resmoe",
        "--variant",
        "resmoe",
        "--base-checkpoint",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_location_residual_1m_total_v2/development_seed101/base/models/sum_single.pt"
      ]
    },
    {
      "method": "resmoe",
      "seed": 102,
      "path": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed102/resmoe_command.json",
      "command": [
        "bash",
        "run.sh",
        "--config",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/configs/location_500000.yaml",
        "--journal",
        "--seed",
        "102",
        "--eval-seeds",
        "1001",
        "--eig-only",
        "--source-check-manifest",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/source_manifest.json",
        "--output",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/location/seed102/resmoe",
        "--variant",
        "resmoe",
        "--base-checkpoint",
        "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed102/base/models/sum_single.pt"
      ]
    }
  ],
  "selection": "Final exact-budget checkpoint; no best-EIG selection",
  "provenance": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_location_routing_scorebed_correction/metadata/pre_edit_source.json"
}
```

## 8. Historical ScoreBED101 slow evaluation explained — PASS

```json
{
  "records": [
    {
      "seed": 101,
      "run": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10132026/10132026_bounded_location_restart/pilot",
      "summary_path": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10132026/10132026_bounded_location_restart/pilot/summary.json",
      "log": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10132026/10132026_bounded_location_restart/pilot/logs/run_log.log",
      "evaluation_seconds": 983.85383937601,
      "training_seconds": 210.57922523096204,
      "forward_particle_stages": 173623525984,
      "secondary_metric": "posterior parameter MSE",
      "mse_inference": "two_independent_tempered_resample_move_SMC_populations",
      "posterior_artifact_count": 125,
      "example_posterior_file": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10132026/10132026_bounded_location_restart/pilot/eval/scorebed_seed1001_posterior_batch16.npz"
    },
    {
      "seed": 102,
      "run": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed102/scorebed",
      "summary_path": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed102/scorebed/summary.json",
      "log": "/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed102/scorebed/logs/run_log.log",
      "evaluation_seconds": 32.49561437801458,
      "training_seconds": 210.74558246089146,
      "forward_particle_stages": 60000060000,
      "secondary_metric": null,
      "mse_inference": "not_run_EIG_only; no MSE value is implied",
      "posterior_artifact_count": 0,
      "example_posterior_file": null
    }
  ],
  "conclusion": "Historical seed101 evaluation_seconds includes high-precisionEIG AND two-replica posterior/MSE diagnostics. It is not EIG-only runtime or score/policy training time. New jobs use --eig-only.",
  "extra_forward_particle_stage_work": 113623465984,
  "timing_limit": "No separate saved stage timer isolates exact EIG and posterior seconds in seed101. Do not subtract cross-run times as an exact decomposition."
}
```

## 9. Failure/partial completion gates and no historical overwrite — FAIL

```json
"AssertionError: "
```

## Controller limitations


These are recovery/supervision limitations, not an observed incorrect training treatment. No fault injection was performed. Current queue continues unchanged. See pairedJSON for full per-seed hashes and commands.
