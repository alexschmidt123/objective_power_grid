# V5 Ubuntu preflight

## 1. Repository & environment

Host: qiqi-workstation. TEMP: `/home/qiqi/Documents/objective_power_grid_temp`. STABLE: `/home/qiqi/Documents/objective_power_grid`. Branch: `development/two-pc-consolidation-20261027`. HEAD: `6a592c8d2ad10eb890f1d351e579ee4129939227` with existing dirty/untracked development changes; complete status is in environment_and_provenance.json. Python3.10.20,PyTorch2.11.0+cu128,CUDA12.8,GPUNVIDIA GeForce RTX 5080. Project directories found: ["/home/qiqi/Documents/objective_power_grid", "/home/qiqi/Documents/objective_power_grid_temp"].

Available `src/research/deepseek_sboed.py` has SHA256 `85069fa88b27e62c82ab2fc326d36ac1aa573b01f5e4ceef2ffe0f4fd38e7573` and the V1-shaped policy counts `{"total_parameters": 8434, "activated_parameters_per_context": 6898, "shared_experts": 1, "routed_experts": 7, "top_k": 3, "expert_feature_dimension": 16, "routed_FFN_width": 8, "shared_FFN_width": 8, "FFN_linear_FLOPs_per_context": 3072, "history_encoder_linear_FLOPs_per_observation": 9728, "router_linear_FLOPs_per_context": 224, "action_head_linear_FLOPs_per_context": 64, "FLOPs_definition": "2perlinear multiplyaccumulate; excludesnonlinearities,normalization,topk,dispatch,simulator,likelihood; not measuredGPUFLOPs"}`. This is not proof of genuine V4 lineage. Expected V4 modules and matching formal checkpoints/config are unavailable; full inventory is in v4_path_inventory.json. The formal10.9825/11.0812 references belong to location-finding T30 according to the supplied provenance, not SIR or power-grid; they were not locally re-evaluated.

## 2. Implementation

BLOCKED. Only scratch audit/report files were created. No V5 source, active config, baseline, simulator, V4 file, checkpoint or launcher was changed. Disabled draft modes are explanatory only, not executable registrations.

## 3. Correctness

V4 strict checkpoint/action/loss/gradient equivalence: NOT TESTED because the required source/config/checkpoint is missing. No V5 unit/smoke tests ran. The CPU-only V1 architecture inventory counts total=8434,shared=1,routed=7,topK=3; these checks do not validate V4.

## 4. Diagnostics

V4 empirical saturation, expert/router gradients, routing exposure, action effects and matched-history counterfactual utility: NOT MEASURED. No EIG superiority or specialization conclusion. The theoretical ceiling is not empirical evidence.

## 5. Resources

Training trajectories=0,contrast trajectories=0,posterior samples=0,simulator calls=0,optimizer updates=0,GPU training seconds=0. Environment queries and source hashing only; measured audit wall/CPU time is in compute_accounting.csv. GPU availability was inspected, not allocated for experiments.

## 6. Performance

New V5 EIG: NOT MEASURED. V4 references10.9825/11.0812 are supplied historical references, not new local results; their rounded arithmetic mean is11.03185 and establishes no significance. Locally found location-vsOED summary records: 0. Matching V4/vsOED current formal case/bank provenance remains unresolved locally; no invented vsOED target. Local SIR T3 rows are preserved separately in local_sir_reference_rows.json: corrected Fixed and old Fixed are distinct, and corrected accelerated DAD remains blocked, not a completed trained method. No SIR score is used as a location target.

## 7. Blockers, promotion and next steps

BLOCKED_MISSING_AUTHORITATIVE_V4_SOURCE_CONFIG_AND_CHECKPOINT. Follow blockers_and_next_commands.md for read-only source/config/checkpoint staging and hash verification. No automatic synchronization, source overlay, promotion, push, LabPC experiment modification or formal training was performed.
