# LabPC validation

End-to-end GPU smoke run on the actual frozen V2 actor, with independent selection/test posterior/noise/contrast banks at stages 0, 14 and 29. Reduced sample sizes and contrasts are for correctness only, not scientific results.

Assertions cover:

- Six mixtures have shape [B,6,4] and normalized nonnegative weights.
- The learned intervention branch matches native V2's complete episode within stated float tolerances at stages 0,14,29.
- Branches preserve their observed prefix exactly and zero unobserved fields before continuation.
- All simulated observations match the exact location-finding signal plus shared prescribed noise, with feasible actions.
- Stage-zero posterior equals the prior: 8,192 proposal particles yield ESS 8,192 and equal importance weights; sample moments agree with the standard-normal prior within the deterministic test tolerance.
- Deliberately conflicting selection/test rankings verify that test rankings cannot alter mixture choices.
- No optimizer steps, unchanged actor state and unchanged historical/source/checkpoint hashes after the smoke run.

Final source revision repeats the smoke test after introducing all-history posterior-quality preflight. Full-run ESS/max-weight thresholds are locked before execution. Passing smoke tests does not imply posterior accuracy at every full-run history or an EIG advantage.
