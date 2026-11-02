# Integrated information-aware MoE

This is a compact adaptation of [DAD (Foster et al., 2021)](https://proceedings.mlr.press/v139/foster21a.html) and [DeepSeekMoE (Dai et al., 2024)](https://aclanthology.org/2024.acl-long.70/), not a reproduction of language-model scaling results.

Architecture: original raw observed action/observation 3-256-ReLU-16 sum encoder, LayerNorm and known stage embedding; identity feature residual plus one always-active width8 shared SwiGLU and three of seven width8 routed SwiGLUs; one shared LayerNorm/linear two-coordinate action head; original sigmoid map to [-4,4]^2. Experts emit features, not independent policies. Total parameters 8434, activated per context 6898. No critic, teacher, PIMoE checkpoint, vsOED action supervision, posterior input, or pretrained weights. Initialization seed101 is fresh.

Offline routing: draw independent standard exponentials E_i and take descending top3 of logits_i-log(E_i). This exponential race equals Gumbel-top-k and samples an ordered Plackett-Luce selection without replacement. For s=(i1,i2,i3),

\[
\log p_\psi(s|h)=\sum_{j=1}^3\left[\ell_{i_j}(h)-\log\sum_{i\notin\{i_1,\ldots,i_{j-1}\}}e^{\ell_i(h)}\right].
\]

The selected feature gates retain the original full-softmax probabilities (selected total mass need not equal one). The full seven-way softmax sums to one. No rescaling to sum selected gates to one is introduced. Ordered sampling is valid even though the feature sum is order invariant: summing ordered outcomes induces the correct unordered-set distribution.

Sample theta0 and L=8192 independent prior contrasts once per simulated trajectory. With prefix cumulative conditional likelihood log a_(t,l), define

\[
U_t=\log a_{t,0}-\log\left(\frac{1}{L+1}\sum_{l=0}^{L}a_{t,l}\right),\quad U_0=0.
\]

Actual code uses logsumexp, with independent nuisance-free location Gaussian likelihood and reparameterized observation noise. The same truth and contrast draws define all prefix values. The terminal pathwise loss uses checkpointed chunks; prefix values are detached. Column j of prefix_information is U_(j+1). Routing stage j (zero based) uses G_j=U_T-U_j, so the previous-prefix vector is [0,U_1,...,U_(T-1)]. No final-stage off-by-one error or repeated immediate reward is used.

The finite-L sPCE objective is the training objective, not exact EIG, and is bounded above by log(8193)=9.0110. The official independent scorer uses 1000000 contrasts; the finite-resolution objectives must not be confused. All simulation and optimization occur offline.

## Gradient identity and history dependence

Let epsilon contain all independent simulator/prior randomness and let S contain all routing draws. With sampled identities held fixed, the correct expectation derivative is

\[
\nabla J=\mathbb E\left[D_\eta U_T+\sum_t\operatorname{sg}(G_t-b_t)D_\eta\log p_\eta(S_t|h_t)\right].
\]

Here D is the total derivative in the reparameterized physical simulation with discrete identities fixed. A selected expert receives pathwise action/information gradients; router weights receive both differentiable selected-gate gradients and a single score term. Future routing probabilities also depend on prior continuous designs through history. This latter derivative is not contained in D U_T when discrete future selections are fixed.

To prevent accidental history credit in the direct conditional score, compute c_t=log p(S_t|sg(h_t)) with the encoder and router parameters still live. Compute q_t=log p(S_t|h_t) using the existing live history, and explicitly add the zero-valued correction q_t-c_t. The direct term has no gradient to physical history. The correction has no extra direct parameter derivative; it restores only the causal history derivative. Combined score is c_t+(q_t-c_t)=q_t, counted once. Detaching history in both terms gives a biased estimator, demonstrated by a two-stage analytic counterexample; exact enumeration plus finite differences validates the corrected estimator. This distinction matters for shared encoder and earlier action-head parameters as well as the router.

Causality permits subtraction of U_(t-1), since its value is fixed before the current selection and conditional expected score is zero. Baseline b_t is a stage-dependent EMA from PREVIOUS optimizer minibatches, initially zero, updated with decay .95 only after the current loss/backward/update. Reward and baseline are stopped in the score term. No current-batch mean including its own sample is subtracted, no gradient propagates through reward estimates in the score term, and no pretrained critic is used.

The maximization objective is implemented as
\[
L=-\overline{U_T}-\overline{\sum_t\operatorname{sg}(G_t-b_t)[c_t+(q_t-c_t)]}+L_{balance}.
\]

The information score coefficient is exactly one. The existing .001 detached-frequency DeepSeek balancing surrogate is retained, with its paper-style stop-gradient semantics; it is a mild regularizer, not an additional information reward or an unbiased estimate of every derivative of an expected batch-balancing statistic. Joint gradients are clipped to norm5 only after all valid information-gradient terms are combined. Raw gradient estimates satisfy the identity; nonlinear clipping and Adam, like the original trainer, do not preserve estimator unbiasedness after transformation. Reward clipping/normalizing or arbitrary per-expert rewards are not introduced.

Per-minibatch direct router score variance is computed analytically from exact PL scores and current detached contexts, reported separately from temporal variability across logged gradients (which also includes parameter drift). Actual shared/router/per-expert norms, selected mass, utilization, contributions and action distributions are logged. Every selected expert has pathwise credit differentiated by its own features and gate; specialization is a hypothesis to test rather than an architectural guarantee.

## Frozen protocol and interpretation

One training seed101, native batch2000, 8192 contrasts, 30 stages, Adam 5e-5 betas(.8,.998), exponential .98 schedule every1000 updates, joint clip5, at most14400 exclusive-device charged training seconds. Readiness and validation are separately counted. Latest compute-eligible checkpoint is selected independently of official EIG; deterministic top3 is predeclared for deployment. No broad sweep, second routing learner, counterfactual loss, or teacher is trained.

Development seed1031 uses matched truth/noise/contrast bank for learned, uniform random subsets (three router-randomness replicates), frozen masks, v1/v2 and completed dense/shared-only/routed-only controls. Uniform preserves learned top3 total mass and divides it equally among a uniformly random subset, matching the historical diagnostic. The mean-of-three uniform comparison's trajectory MC SE does not fully measure uncertainty over all possible router seeds. Historical controls use lower training contrast resolution; same four-hour cap does not make training throughput or objective resolution identical. No v3 run is started, and its missing checkpoint is explicit.

Limited optional fixed-history swaps use truth-excluded prior importance weights from actual observed history and independent posterior-predictive fantasies; low-ESS histories are rejected. They estimate one-step conditional information under a finite-particle approximation, not full remaining-horizon EIG. ESS alone is not proof of posterior convergence. No hidden simulated true theta is used as a posterior distribution and no swap measurement trains the router.

After numerical freeze, official seed1001 uses2000 cases/L1000000/T30 and the original independent location scorer/canonical bank, even on poor development performance. This is a consumed historical benchmark. Seed1079 is reserved before training following a saved metadata prior-use scan; its bank is generated/sealed before training and scored only after freeze. Paired v1 comparisons reuse identical bank/noise. One seed does not establish multi-seed superiority, and the target12.48085 is aspirational. Negative results remain results.

Deployment is one neural forward call from observed history, followed by the original feasible-map. No simulation, EIG estimator, candidate search, optimizer, counterfactual, lookahead or parameter update is part of policy.forward. Automated mocks fail if core forbidden routines are called. Conditional diagnostic simulation is offline only.
