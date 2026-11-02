# FH-MoE-sBOED-v3: router-independent full-horizon context specialization

Status: context discovery is implemented; specialization and full training are conditional on the predeclared discovery/scientific gates. This is a proposed training correction, not demonstrated EIG superiority.

## Foundation and scope

The foundation is exclusively DeepSeekMoE-v1 commit `13681d7091189a4e68056e6a2864dc58f9d5700b`. The original policy, trainer, simulator, rollout, DAD encoder and evaluator files remain byte-identical. Its validated checkpoint SHA256 is `f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7`; configuration SHA256 is `b2c920b9830babf3cec75a6fc6c0c305e5ae1a1104744d3c9f368101f360d8b0`, independently verified on LabPC. Historical canonical EIG is 10.98906590 +/- 0.04479411 nats. The checkpoint is not initialization for this pilot or a future primary model.

The eventual deployed policy remains the 8434-parameter v1 model: DAD sum-history encoder, one shared width-8 SwiGLU expert, seven routed width-8 SwiGLU experts, top-three sparse routing and one common head. Selected full-softmax scores retain the original unnormalized top-three mass. No density model, critic, teacher policy or online planner is introduced.

## Why v2 could not establish contexts

V2's reference update was an exponential average of unit terminal-loss adjoints weighted by existing expert gate exposure. Thus each purported context was identified with an expert slot before context differentiation had been demonstrated. Overlapping routing populations can generate similar mean directions. The references also combined gradients from changing policies; a stage was revisited every 40 policy updates, and decay .95 retained a long window over policy changes. Split-stream agreement did not fix this bootstrap dependence. In the saved pilot, only one auxiliary update qualified and no stage supplied usable held-out targets. These are supported implementation observations; exposure bootstrapping is a plausible failure mechanism, not proof of a unique causal explanation. The previously corrected CPU/default-CUDA allocation and retained-graph OOM errors are distinct implementation failures, not evidence that all gradient-context methods fail.

## Router-independent reference

Initialize the original full policy from scratch. During a bounded preparation phase, set its existing routed masking switch to true. Train only the shared path and backbone with the native terminal loss, Adam, clipping and update-based schedule. A matched native-v1 control starts from exactly the same state and receives the same simulation seeds, batch size, contrasts and preparation update count. The reference's changed path is an offline discovery instrument; its EIG is not called a v3 specialization result.

Freeze this reference before collecting discovery data. The shared path is
\[
 z_t=\mathrm{encode}(h_t),\qquad u_t=z_t+E_s(z_t),\qquad a_t=\mathrm{feasible}(H(u_t)).
\]
No routed FFN or router is executed. Unit tests alter all routed weights and the router arbitrarily and verify identical reference actions and covariates.

## Terminal learning demands

For complete simulated trajectories and the original L=2000 contrasts,
\[
 \ell_i=-\left[\log p(y_{1:T}^{(i)}\mid\theta_0^{(i)},a_{1:T}^{(i)})-
 \log\left({1\over L+1}\sum_{j=0}^L p(y_{1:T}^{(i)}\mid\theta_j^{(i)},a_{1:T}^{(i)})\right)\right],
 \qquad d_{it}=B\,\partial\left(B^{-1}\sum_i\ell_i\right)/\partial u_{it}.
\]
Autodifferentiation includes the full later action-observation-history graph. This is a terminal-loss adjoint, not intermediate information gain. Capturing tensors does not modify or detach native policy gradients. Finite differences perturb an intermediate pre-head feature and recompute all 30 stages. Per-trajectory gradients are normalized only for statistical discovery: \(g=d/\max(\|d\|,10^{-30})\). Raw adjoints are saved for numerical sensitivity checks. No gradient grouping changes the native rollout or sPCE formula.

## Observable conditional demands and independent replication

At each predeclared stage t in {1,10,20,29}, use 41 offline covariates: frozen reference encoding z, coordinate squares z^2, mean and root-mean-square of the observed normalized design/observation triples, and the latest observed triple. Fits are stage-specific; both stage and remaining horizon are known, never latent. The covariates do not contain theta, contrast identities, simulator state, future observations or routed indices.

Three independent simulation splits each contain 8000 complete trajectories. Fit a standardized ridge conditional-mean estimate on split A only:
\[
 \widehat W=\arg\min_W n^{-1}\sum_i\|g_i-\phi(h_i)W\|^2+.01\|W_{\rm nonintercept}\|^2.
\]
Fit the same model independently on split B. The model is an offline regression of learning demands, not a deployed teacher or an information critic. Standardization uses each fitting split only. Deterministic two-component spherical clustering uses the conditional predictions from A, with no expert labels. On untouched split C, soft context responsibilities are
\[
 r_{ik}=\mathrm{softmax}_k\left(\mathrm{normalize}(\phi(h_i)\widehat W)^\top c_k/.25\right).
\]
Two components are a bounded scientific hypothesis, not a forced discovery: collapsed or unreliable structure fails the gate. There is no requirement for seven contexts.

## Predeclared discovery gate

Each accepted stage must satisfy all checks: held-out squared gradient-prediction improvement over the constant training mean (upper normal-approximation 95% interval below zero), at least 10% test occupancy per context, split-half mean-direction cosine >=.8, mean unit-gradient norm >=.1, mean-gradient signal-to-noise >=3 in both halves of each context, between-context mean cosine <.8, replicated assignment Cohen kappa >=.6, and >=99% float32/float64 assignment agreement. At least two stages must qualify. Full-graph network precision sensitivity is checked separately with the canonical float64 simulator. All gates and seeds were serialized before GPU data collection. Intervals are exploratory case-level summaries, not familywise-confirmatory scientific tests.

Contexts are evaluated on joint-distribution histories. They are not exact fixed-history posterior classes, and no generating theta is treated as a posterior for an observed history. Normalized-adjoint predictability is necessary evidence for the proposed mechanism, not proof of useful information-acquisition competence.

## Conditional next stage

Only if discovery passes may its validated soft history responsibilities differentiate existing routed feature modules and supervise the original internal router. Any first-order terminal-loss auxiliary must be explicitly a conditional scalar surrogate with fixed responsibilities and adjoints; live encoder/history derivatives and the native gradient must be retained, with the native loss differentiated once. It must not be described as an unbiased estimator of the unmodified EIG gradient. The complete MoE must then pass a matched native-v1 development EIG/intervention gate before primary training. No labels, specialist modules or successful optimization are asserted when discovery fails.

## Source and literature

Implementation: `src/research/fh_moe_v3_context.py`, `src/research/fh_moe_v3_discovery.py`; correctness tests: `tools/tests/test_fh_moe_v3.py` and original `tools/tests/test_deepseek_sboed.py`.

[DAD, Foster et al. 2021](https://proceedings.mlr.press/v139/foster21a.html) supplies the sequential contrastive training objective and amortized history-to-design approach. [DeepSeekMoE, Dai et al. 2024](https://aclanthology.org/2024.acl-long.70/) motivates shared/routed feature decomposition, studied there in language models. Neither paper establishes the proposed router-independent gradient-context mechanism or its usefulness for this location benchmark. No novelty or superiority claim is made.
