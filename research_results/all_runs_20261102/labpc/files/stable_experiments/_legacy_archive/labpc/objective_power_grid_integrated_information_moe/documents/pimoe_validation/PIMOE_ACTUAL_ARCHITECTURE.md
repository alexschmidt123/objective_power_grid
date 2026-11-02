# Actual frozen PIMoE architecture

Authoritative scientific implementation:75f74ed29e9f87cfb95bc80e6cd28d36f8573c25; `src/research/pimoe.py`, classes `DensityHead`,`PredictiveMoE`,`AdaptiveBelief`,`PIMoEPlanner`, functions `state_features`,`predictive_loss`. Training phases: `src/research/pimoe_experiment.py::main,train_model,dataset`. Validation preserves those files byte-for-byte.

Four independent experts, each Linear(10,48),tanh,Linear(48,48),tanh,Linear(48,9). The9 outputs give3mixture logits,3location offsets and3scale logits. There are3321parameters per expert, no shared expert parameters. Router is Linear(11,32),tanh,Linear(32,4),516parameters; total13800. No trainable history encoder, actor, critic, score network or direct design expert exists.

For posterior particles theta_p and candidate a, true simulator gives mu_p(a). Nine3x3physical-box anchor responses, standardized by posterior sample mean/std plus sigma, form9router features; candidate response standardized by posterior-weighted mean/spread adds1; stage/30 adds1. The router never sees raw theta or true latent parameters. Source exchange leaves all responses invariant. Softmax over4experts gives r_ape. Assignments remain SOFT during training and planning; no ST/DSelect gating is used by PIMoE. No expert is sampled for action execution.

alpha_ae=sum_p w_p r_ape; c_ape=w_p r_ape/alpha_ae. Region response moments m_ae,v_ae,skew_ae summarize all posterior particles with these weights. Each head receives exactly10features: regionmean/3, log(sqrt(v_e+sigma²)), normalized regional third moment,alpha_e,globalmean/3,log(globalspread),global third moment,actionx/bound,actiony/bound,stage/30. Anchor features normalize with unweighted sample moments; after resampling posterior weights are uniform. Tiny mass denominator is clamped1e-12 for numerical safety.

Expert component weight pi_e=softmax(logits); location=m_e+sqrt(v_e+sigma²)*offset; scale=sqrt(sigma²+(sqrt(v_e+sigma²)*softplus(scale_logit))²). Each q_e is a3Gaussian mixture. q_mix=sum_e alpha_e q_e is normalized with12total Gaussian components. Predictive mean/variance differ through both regional posterior inputs and independently learned head functions.

Every counterfactual observation trains the complete mixture by NLL. There is NO separate expert-specific regional NLL loss despite an informal source comment. Different learning signals arise from differing region-conditioned inputs and mixture responsibilities alpha_e q_e(y)/q_mix(y); low-probability experts receive weaker predictive gradients. Full mixture entropy/ranking gradients also couple all heads and router. Router gradients additionally pass through regional moments and mass.

Exact loss: mean[-log q_mix(y)] +0.1mean_confident_pairs softplus[-sign(Iref_i-Iref_j)*(Ihat_i-Ihat_j)] +0.02mean_a[sum_e alpha_e Var_e(mu)/(Var_w(mu)+sigma²)]. Confident pairs require reference separation>2combinedMCSE. The final term affects routing/region formation, not density-head parameters directly. No explicit equal-load, entropy-balancing, diversity, or anti-collapse loss exists. Source separation labels and exploration/exploitation roles are not predefined.

Reference entropy is estimated with independent observation RNG and posterior-stratified outer draws against the full weighted Gaussian predictive mixture of the approximate SMC belief. Predictive NLL targets use separate draws. This is not sPCE training. Predicted EIG=H(q_mix)-0.5log(2pi e sigma²), evaluated using24-node Gauss-Hermite per mixture component. Approximate-density entropy is not a guaranteed EIG bound.

Adam0.001 jointly trains router+all experts; clip global norm5. Source has2000pilot updates then12000follow-on updates, with Adam state REINITIALIZED at the phase boundary. Replayed data are24posterior states from6complete exploratory histories, plus12held-out states from3histories,8counterfactual actions/state. Original single control previously received only2000updates; this study adds its matched12000phase. Last fixed-schedule checkpoint, not highest-EIG selection. Source random training-data generation uses1004/1005; official1001 is not used for model training/selection.

Planner:24random feasible candidate locations,3local refinements, immediate predictive entropy scoring, shortlist2,2fantasy observations per shortlisted action, full belief update per fantasy and best second-step score. At step30 immediate selection only. SMC uses2048particles for training,4096in official planning, adaptive tempering, resampling, eight MH moves/bridge and source-label swap symmetry. CPU inference/refinement dominates online work; one published139ms selection excludes the actual post-observation posterior update, which this validation times separately. No30-step global-optimality claim.

Facts above are verified code behavior. Meaningful and reproducible specialization is a hypothesis requiring output/calibration/contribution/controlled-EIG evidence. Independent initialization, occupancy, or distinct weights alone are insufficient.

## Source line traceability

- `src/research/pimoe.py:13` `rng` (frozen original commit75f74ed).
- `src/research/pimoe.py:16` `conditional_entropy` (frozen original commit75f74ed).
- `src/research/pimoe.py:21` `responses` (frozen original commit75f74ed).
- `src/research/pimoe.py:27` `count_work` (frozen original commit75f74ed).
- `src/research/pimoe.py:31` `AdaptiveBelief` (frozen original commit75f74ed).
- `src/research/pimoe.py:104` `DensityHead` (frozen original commit75f74ed).
- `src/research/pimoe.py:115` `PredictiveMoE` (frozen original commit75f74ed).
- `src/research/pimoe.py:157` `state_features` (frozen original commit75f74ed).
- `src/research/pimoe.py:161` `direct_reference` (frozen original commit75f74ed).
- `src/research/pimoe.py:175` `predictive_loss` (frozen original commit75f74ed).
- `src/research/pimoe.py:190` `PIMoEPlanner` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:12` `write` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:15` `sha` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:17` `dataset` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:43` `fitted_evaluation` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:57` `train_model` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:71` `posterior_validation` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:85` `final_evaluate` (frozen original commit75f74ed).
- `src/research/pimoe_experiment.py:103` `main` (frozen original commit75f74ed).
