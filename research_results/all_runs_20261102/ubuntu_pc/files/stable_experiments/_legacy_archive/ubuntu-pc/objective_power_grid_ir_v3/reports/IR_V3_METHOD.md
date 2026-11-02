# IR-MoE-sBOED-v3 method and fixed pilot protocol

One shared expert, seven routed width8 SwiGLU experts, top3, shared raw3-256ReLU-16 sum encoder, shared head,8434 total/6898 activated parameters. No RC gain, teacher, external router, or architecture change. The prior, simulator, observation likelihood, bounds[-4,4],H30,B2000,L8192 and four-hour main allowance are inherited unchanged from v2.

## Conditional references

Targets are pi_beta(theta) proportional to standard-normal prior(theta)*p(observed history|theta)^beta. Adaptive temperatures maintain incremental ESS>=0.8N. Every bridge uses systematic resampling and8 Metropolis-Hastings rejuvenation steps. Fixed, state-independent symmetric Gaussian proposal scales are0.02,0.08,0.3,1; MH uses the exact prior+tempered-history likelihood ratio. A50% exchange of the two identical source labels preserves the target. No generating hidden theta enters a reference or deployed policy.

Independent N4096/N8192 references are checked using16 independent predictive design probes. Maximum normalized predictive-mean difference<=0.15, normalized variance difference<=0.25, and mean MH acceptance>=0.05 are numerical consistency requirements. Early references are additionally compared with independent262144-particle importance references when their ESS>=128. These are practical validation screens, not convergence theorems. Uniform weights after resampling do NOT imply independent posterior ESS=N; initial importance ESS, ancestry and MH acceptance are retained.

## Paired variance reduction and independent confirmation

For each identical observed prefix, compare the learned top3 with one predeclared single-expert swap. Both worlds use the same posterior draws, contrast draws, future observation noise and stochastic PL-continuation randomness. The same frozen MoE supplies the continuation. No deployed design search occurs.

For each independent cluster, average paired information differences under epsilon and -epsilon. Both noises have the correct Gaussian marginal. The estimator is unbiased for the same finite-reference conditional utility; variance reduction is not guaranteed and is measured. Treat the antithetic PAIR as one cluster when estimating MC SE, never as two independent observations.

Adaptive screens use16,64,256 clusters per reference and allocate independent confirmation to32,128,512 clusters. Screen draws are never reused for inferential labels or the live gradient. One fixed-size confirmation follows each screen, without optional extension after inspecting it. Both posterior groups separately require a same-sign Student-t interval outside[-0.005,0.005] nats. Bonferroni covers8 probes*2 reference groups at family alpha0.05. Student-t MC intervals are a normal approximation for actual information utilities, not a finite-sample nonparametric guarantee, and condition on empirical posterior references. They do not remove posterior bias.

Pilot:8 independent histories,2 each at stages1,5,15,25; maximum600 GPU wall seconds and50billion parameter-stage propagations. Training requires>=6 numerically consistent references and>=2 independently confirmed practical advantages in>=2 stages. No statistical threshold is weakened after results.

## Integrated gradient

Primary loss remains negative terminal full-horizon sequential sPCE plus the correct PL score-function loss and mild0.001 balancing. Score coefficient is ONE. Terminal pathwise gradients remain intact. Stage reward-to-go is U_T-U_(t-1), with U_0=0, same contrast bank at all prefixes, stopped advantages and previous-update stage EMA baseline.

For the complete hybrid estimator, E[d_cont U + sum_t stop(G_t-b_t)d log p(S_t|h_t)] equals the expectation gradient. The full conditional log-probability derivative includes earlier-action/history paths. The inherited conditional-logit plus zero-valued full-minus-conditional correction preserves them without duplicating current encoder/router credit. Both the original and antithetic conditional estimator agree with exact two-stage enumeration and finite differences on CPU and CUDA.

For admitted fixed reference data, use w[-log sigmoid(log P(winner|h)-log P(loser|h)) + negative conditional information surrogate], with w given by the conservative lower-confidence excess beyond the practical threshold, capped0.1. Unordered set probability sums all six ordered PL probabilities. Selected expert features/actions remain live; later continuation weights are frozen, while necessary history-routing derivatives remain. Independent live fantasies prevent fitting lucky confirmation noise. Reference weights, reward estimates, labels and the decision gate are stopped. The auxiliary is explicitly a distinct frozen-context improvement surrogate, not a second unbiased estimate of the primary objective. Existing maximum coefficient0.05, auxiliary-gradient fraction0.1 and global norm5 clipping remain.

Deployment: frozen history encoder -> frozen router -> shared+top3 routed features -> frozen head -> feasible experiment. No posterior sampling, MC probes, teachers, planning, candidate search or parameter updates online.

Sources: [DAD](https://proceedings.mlr.press/v139/foster21a.html), [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/), [SMC samplers](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf), [antithetic/common random numbers](https://pubsonline.informs.org/doi/10.1287/mnsc.21.10.1176). V3 is a new proposed adaptation, not a demonstrated contribution of those papers.
