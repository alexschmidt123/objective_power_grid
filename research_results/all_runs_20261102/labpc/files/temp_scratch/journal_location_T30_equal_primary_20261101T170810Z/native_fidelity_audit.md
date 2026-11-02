# Native fidelity / readiness audit

No production common budget is selected. No new EIG result exists.

| Method | Implementation and allocation | Current limitation |
|---|---|---|
| Random | Canonical truncated-standard-normal feasible design distribution; no training | Bank evaluation pending |
| Fixed | Canonical joint terminal sPCE; batch64, L1024, Adam5e-5, betas(.8,.998), clipping5 | Four-update timing probe is not optimization-quality evidence; robust initialization/selection protocol needs freezing and exposure accounting |
| DAD | LocationDAD native history encoder, full-horizon sPCE, batch2000 L2000, Adam5e-5, no extra clipping | Exact continuation RNG/optimizer equivalence not yet tested |
| RL-sBOED | Pinned native SAC/REDQ, batch100 complete paths, L100000; replay4096, 64 optimization cycles after100000 transition warmup | Existing final rl_sboed.pt has actor only. Replay, critic/targets, optimizers, temperature/RNG are required for exact resume |
| native vsOED | Pinned GMM-I actor/critic/posterior; initial20000+1000 per cycle, replay10000, five critic/posterior updates, discount.9 | Vendor pg_vsoed.py lines550–564 unconditionally constructs LOCAL replay. Saving PGvsOED alone loses this buffer; adapter final checkpoint also excludes target/optimizer/RNG state |
| ScoreBED | Pinned JAX score matching then score-gradient policy training; P1 one restart; initialization10014, budget-adapted95:5 remaining paths | Training-state msgpacks omit complete trainer RNG/pipeline position; restart reconstructs standardization. Native LR reference horizons must stay190000/10000; partial-tail and phase-adequacy validation pending |
| MoE-V1 | Historical DeepSeekPolicy full,8434 params, shared1/routed7/top3; balance.001; native full-horizon sPCE | Original V1 reference, NOT V5 OFF rebranded as novelty. Full recovery continuation test pending |

The CPU mathematical tests check the truth-inclusive terminal denominator, feasible T30 actions, architecture cardinality, and rejection of duplicate/incomplete primary path IDs. These do NOT certify all native counters. Native replay should be counted as computation, not new exposure. Full continuation test must compare an uninterrupted run with save/load/resumed run on identical RNG state, including subsequent draws, losses, and all updated parameters.

The present failure is an implementation-readiness gap, NOT proof that 1M/2M is physically infeasible or that native objectives must be replaced. No shorter surrogate production run is authorized by this report.

Selected current-policy tests: `profiles/current_policy_gradient_tests.json` reports all three full-T30 finite-difference checks PASS and strict state/forward roundtrip PASS. `logs/cpu_regressions.log` retains the two retired-ID failures. All source-integrity hashes still match the initial snapshot.
