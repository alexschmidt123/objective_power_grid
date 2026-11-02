# Controlled SIR baseline correction, development seed 101

Historical ten-method seed101 campaign is complete and immutable. New runs use the same physical SIR model, T=5, primary budget exactly 1M, final-budget selection, same optimizers and native objectives. Evaluation seed1001, 2048 trajectories/L16384 remains development, not precision-certified confirmation.

ScoreBED: one SIR-specific native option, potential_hidden_dim=128, as in the pinned gravimetry preset. This adds a native potential projection before LayerNorm to avoid its documented two-feature gradient degeneracy. Transformer, native score objective, policy architecture and budget split remain unchanged: 10014 initialization, 692990 score, 296996 policy. This is a disclosed benchmark adaptation, not an exact published SIR reproduction.

vsOED: one deterministic feature-unit adaptation: design percentages divided by100 and infected-count inputs divided by500 at actor, target actor, critic, target critic and posterior feature networks. Physical simulator, percentage output bounds, exploration, likelihood, variational rewards, replay and optimizer settings remain unchanged. Gradients through scaling are retained. Historical adapters stay default unless enabled. The initial-input probe supports saturation as a failure mechanism; it does not prove scaling will improve converged EIG. No extra trust region, clipping or learning-rate change is added.

Launch through canonical run.sh, sequentially to avoid GPU contention. Preserve all failed/partial attempts; no automatic overwrite. No other methods are retrained. LabPC queue is untouched. No final evaluation seed is used. Source and configuration snapshot/hash checks precede full runs.
