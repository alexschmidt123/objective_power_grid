# Status and next reproducible actions

V4 provenance/parity blocker is CLOSED for the recorded Ubuntu CPU scope. V5 diagnostic and gated objective/credit components are implemented and tested, not a validated trained method.

Remaining scientific blockers:
1. V5-A: a calibrated location posterior sampler respecting source-label symmetry, multimodality, past-only conditioning, ESS/coverage/mixing/proposal validation and explicit sampling-gradient approximation. Gaussian toy correctness is not location validation.
2. V5-B: utility resolution and independent replication. The mandated tiny L32 smoke produces all-zero terminal differences in the sampled cases. High-resolution matched-history utility is NOT MEASURED, so no specialization or noise-filtering benefit is established. No forced expert-exposure rule was enabled.
3. No trained V5 checkpoints, paired V5/V4/vsOED comparison or seed101–105 replication. No formal training action exists in this diagnostic launcher. A costed native-training integration and separately authorized run plan is required, including all posterior/counterfactual overhead within the14400-second reference cap.
4. Combined remains blocked until both single-change scientific gates pass. Promotion NOT APPLIED; no GitHub push.

Actual tests are in exact_test_command.txt and unit_test_output_final.txt (23passed). For a new, non-overwriting diagnostic-only repeat:

```bash
cd /home/qiqi/Documents/objective_power_grid_temp
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 bash run.sh --moe-sboed-v5-development --config configs/moe_sboed_v5_diagnostics.yaml --seed 101 --output scratch_runs/10312026_054200_moe_sboed_v5_ubuntu/recheck_diagnostics_seed101_001
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 bash run.sh --moe-sboed-v5-development --config configs/moe_sboed_v5_expert_credit.yaml --action credit-smoke --seed 101 --output scratch_runs/10312026_054200_moe_sboed_v5_ubuntu/recheck_credit_seed101_001
```

These commands are diagnostic-only, not performance tests. No command for a nonexistent formal V5 trainer is supplied. Preserve reference hashes and do not overwrite existing outputs. Future publication and GitHub operations remain LabPC-only and require authorization.
