# Corrected IC budget decision

Original PIMoE freeze: 1210.631910391152 charged experimental seconds. Its measured optimizer phases totaled 70.91861508367583 seconds. The IC implementation introduced the latter as an additional cumulative optimization ceiling; no original evidence supports a per-valid-attempt reset.

The requested fresh corrected attempt receives an explicitly declared 70.91861508367583-second attempt ceiling. This exceeds the previous cumulative optimization restriction and MUST NOT be described as satisfying that restriction. The unchanged cumulative experimental ceiling is enforced through the SAME persisted ledger, including invalid training, tests, data generation, and corrected training. No clock is reset. Independent official evaluation is reported separately. No pretrained invalid weights are loaded.

Rare-expert bug: a fixed denominator floor in ESS=(sum r)^2/sum(r^2) broke scale invariance and yielded ESS below 1. Corrected log-domain membership normalization and ESS=1/sum(normalized_r^2) preserve finite normalized conditional regions. A regression explicitly reproduces the prior error. Architecture and loss coefficients unchanged.
