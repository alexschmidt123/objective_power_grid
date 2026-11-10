# Router and responsibility rule: valid foundation, missing additional mechanism

INCOMPLETE: no scientifically selected v3 training rule.

For bounded pi_k(a|h), g_k(h)>0 and sum g=1, sample K~g, a~pi_K, then simulate a new observation and reroute. Posterior component responsibility is r_k=g_k pi_k / sum_j g_j pi_j. It measures how well expert k explains the sampled action under the mixture. It is not P(expert k is information-optimal|h).

Joint latent-label score credit is A(h,a)[grad log g_K+grad log pi_K]. Marginal credit uses A sum_k r_k[grad log g_k+grad log pi_k]. With a K-independent environment return and baseline, the latter is the conditional expectation of the former given h,a. They have the same expectation; one assigns sampled-label credit, the other can spread action credit to overlapping policies. This is standard mixture policy learning, not a new responsibility target.

The sequential Q must integrate all subsequent routing and experimental observations. Critic-at-mean-action, one-step posterior spread, and one realized noisy suffix are not interchangeable with this value. Additional powers of s_k or winner labels need a derivation, not an appeal to probabilistic responsibility.

An early exploration mixture b=(1-epsilon)g+epsilon/4 is valid if log b, derivatives, data collection and target expectations use b consistently. A time-varying epsilon schedule changes the policy being optimized; gradient/evaluation policy definitions must state it. Using log g on b samples is not automatically on-policy. No schedule was selected or run.

For a genuinely stage-only control, g_B(h)=softmax(l_B(t)), and no belief, condition scores, Q table, shared history feature, cached conditional prior or mixture responsibility may feed its forward router logits. The experts still see full history. Native policy gradients may naturally give stage parameters different training signals across histories; that does not make their deployed gate history-dependent. The two arms must match all other learning opportunities.
