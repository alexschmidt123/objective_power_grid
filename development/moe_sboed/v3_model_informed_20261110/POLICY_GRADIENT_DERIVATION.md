# Native full-horizon mixture gradient

Native incremental variational rewards r_t=log q_w(theta|h_{t+1})-log q_w(theta|h_t),with prior at empty history and the exact terminal convention,are unchanged. Gamma1 telescopes to the terminal variational information score. This remains a variational approximation,not exact MI;no sPCE training or new reward.

For fixed replay occupancy and critic Q,F=E_{h,epsilon}sum_k g_k(h,mu(h))Q(h,a_k(h,epsilon_k)). Let q_k=Q(h,a_k),b=sum stop(g_k)stop(q_k),K~g. Minimize

    L=-Q(h,a_K)-sum_k g_k stop(q_k-b).

The expected selected-action derivative gives sum g_k grad Q_k;the finite gate term gives sum(q_k-b)grad g_k,including proposal-feature paths. Since sum grad g=0,the baseline cancels. Explicit independent Gaussian draws for each expert and categorical uniforms are checkpoint inputs,so recomputation does not redraw them. A finite-sum gradient identity tests the full proposal-dependent gate chain.

The on-policy score equivalent sums A_t[grad log g_K+grad log pi_K] over all30 decisions;posterior component responsibilities yield the marginal-score identity. These are established identities. Native averaging over B*T preserves its scale convention. Learned Q and off-policy replay history occupancy retain approximation bias;no exact/unbiased MI claim.

Bellman targets integrate all four target experts,with one reparameterized continuous sample per expert and target probabilities. Q at a mean action is NOT substituted for the stochastic expectation. Native critic/posterior Adam,schedules,replay and Polyak.1 are retained. Compatible encoder parameter objects/moments survive;new expert/router moments are fresh. Retired incompatible HEC actor state is not falsely copied. Target encoder EMA is preserved;new target heads match preparation;both use the same immutable reference.

For a=4tanh z,log pi_k=log Normal(atanh(a/4);mu_k,.025)-sum[log4+2(log2-z-softplus(-2z))]. Clipping cannot be substituted for a density. External Gaussian exploration is zero for both arms because stochastic sampling is now inside the policy;the .05 categorical exploration is included consistently everywhere. There is no ongoing preparation objective or additional loss.
