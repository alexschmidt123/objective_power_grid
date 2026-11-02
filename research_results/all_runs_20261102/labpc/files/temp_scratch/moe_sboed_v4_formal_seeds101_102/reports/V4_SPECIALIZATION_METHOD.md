# V4_SPECIALIZATION_METHOD

# V4_UTILITY_ROUTING_METHOD

Original loss -mean terminal sequential sPCE +.001 router balance. Every100updates:64fresh joint histories,1024contrasts; one random decision per case and one selected/unselected replacement. Two full paired rollouts use identical truth/noise/contrast draws and identical observed prefixes. Only the selected decision is intervened; later policy resumes. Realized terminal advantage is detached,RMS-normalized,clipped[-1,1]. Router pairwise logistic loss favors replacement for positive utility and original for negative utility. Its additive gradient norm is capped at5% of primary router-gradient norm. This is biased/noisy offline credit, not exact conditional information-to-go or an unbiased EIG gradient. No auxiliary changes to expert/encoder/shared/head gradients; routed experts train through original terminal loss. No arbitrary context-gradient or role loss retained.

The formal mechanism is identical to seed1; no loss/coefficient tuning. Every100updates, B64,L1024 controlled one-swap paired full-horizon continuations; detached RMS-bounded logistic ranking, additive router gradient capped at5% primary router norm. All other parameters retain original full-loss gradients. Numerical safeguards logged;10consecutive invalid credit events stop as numerical/mechanism failure rather than silently train V1.
