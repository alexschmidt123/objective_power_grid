# Executable audit

Active/frozen previous source calls StrategyManager; no bypass was found. Previous ST used actual.detach() + soft-soft.detach() and mixed previous hard-forward state into the regularization recurrence. Thus the sigmoid b weighted current-router derivatives even when hard continuation was requested. Corrected ST uses proposal_ST and switch_ST separately, q=(1-switch_ST)*q_previous+switch_ST*proposal_ST, plus an independent relaxed recurrence for regularization. No state detachment within episodes.

Informative analytic tests prove nonzero switch gradient when proposed/previous experts differ and prove continuation blocks the current proposal derivative while preserving earlier-state gradients. Forward equality alone cannot distinguish these VJPs. This is a confirmed surrogate mismatch, not proof that it caused the old EIG regression. Identical hard experts correctly give zero switching gradient.

63 CPU tests passed, 1 optional historical test skipped on LabPC. Includes T30 actions/bounds/likelihood/gradients, forced reroute/continue, reset, request/change accounting, residual-disabled base equality, checkpoint round trip and portable common evaluator. All four architectures retained. Native budget and frozen-base equality are checked again on each full training run.
