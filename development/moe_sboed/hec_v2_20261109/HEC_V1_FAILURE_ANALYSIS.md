# Frozen HEC-v1 investigation

The saved same-bank result is +0.07211 nats over its matched-update heterogeneous control, CI [0.01889,0.12532], but only E1/E3 showed distinct designated task advantages. No individual removal penalty survived correction; stage-only commands and uniform coordination remained competitive. Saved code/logs/checkpoints and the prior runtime audit were inspected; no v1 training was repeated.

New inexpensive probes use 32 saved held-out histories at stages 3,12,24. A +0.25 perturbation of each expert's own command changes its features and the action. Feature RMS changes E1/E2/E3/E4: 0.00861,0.04569,0.63295,0.02366; action RMS 0.00229,0.01588,0.02505,0.03332. Stage-mean-command action RMS is 0.02503 versus 0.07618 for zero commands. These are functional effects on identical histories, not counterfactual EIG rollouts or proof of competence.

Held-out normalized task-2 probe losses are E2 0.53190 versus E4 0.52347. Task-4 losses are E4 0.62013 versus E2 0.63104, consistent with its inconclusive advantage. Early-stage CKA includes E2/E3 0.720 and E2/E4 0.687, but later overlap is lower: no complete-collapse claim is supported. Raw results are evidence/FROZEN_V1_DIAGNOSTICS.json.

Verified implementation limitations: E2 pools one softly weighted hypothesis representation and predicts marginal moments; it has no explicit opposing-hypothesis comparison. E4 consumes existing scalar-critic probe differences and low-order predictive moments, with a noisy one-step bootstrap task. It has no explicit model-update/future-design feature. Both may be redundant with other features or the strong provisional SG policy. These are plausible explanations, not experimentally isolated causes. Non-significance does not establish redundancy or equality.

V2 therefore adds paired nonlocal likelihood comparisons and explicit predictive posterior-reweighting/future-design features, with independent learned target extrapolation. E1/E3 functions are preserved as available fixed feature anchors. Pure EIG gradients learn how anchors and responses cooperate. No diversity penalty or responsibility teacher is repeated.

Stage-only commands still yield history-dependent expert features because the approximate belief, exact likelihood queries and common SG provisional design depend on observations. Thus competitive stage commands need not imply total expert collapse; they show that the extra learned history-conditioned command variation has not yet demonstrated utility beyond those existing adaptive pathways.
