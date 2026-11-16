# Publication-candidate protocol

Both MoE-sBOED and MoE-sBOED-uniform receive2000 native actor updates per seed,10000 critic updates,300000 posterior updates,2M new T30 episodes and60M observations. Uniform is trained independently with weights fixed at.25. Seed101 initial actor weights match exactly. The four-hour total cap is a maximum;fixed terminal schedules are not selectively extended using test EIG. No pilot/smoke result counts toward the five-seed table.

All final policies use the exact original2000-case/1M-contrast bank(seed202611099206). Internal objectives remain native full-horizon variational EIG. Shared initialization means continuation robustness,not independent from-scratch pretraining. Inherited cost is disclosed. External baseline training is reused;remaining compute differences are reported,including longer historical RL training if reused.

Extracted E1-E4 execute one current expert with common context and no current router/other current-expert features. The common frozen provisional-design provider is itself inherited and may contain historical collaborative computations;this is disclosed rather than claiming complete information isolation. No additional expert training is performed.

Formal locked execution is not a convergence or superiority guarantee. Publication conclusions require completed five-seed results,paired uncertainty,resource audit and verified evaluation/checkpoint provenance. Negative results are retained.
