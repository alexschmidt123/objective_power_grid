# PIMoE-sBOED: predictive-identifiability prototype

Status: experimental, unverified scientific value. One algorithm, one seed; not a renamed policy MoE. No established novelty priority claim.

## Physical problem and Bayesian belief

Import the existing LocationFinding unchanged: two2D sources, four iid standard-normal coordinates, T30, designs[-4,4]^2. Mean log(0.1+sum_i1/(1e-4+||theta_i-a||^2)); independent additive Gaussian noise sigma0.5. The conditional entropy is exactly0.5log(2pi e sigma^2), independent of theta and action; noise does not pass through a log afterward.

Belief starts at the continuous prior and uses the true cumulative likelihood. Adaptive likelihood tempering controls ESS, systematic resampling is followed by full-history Metropolis rejuvenation with fixed-within-kernel covariance, local scales and prior independence proposals. Random source-label swaps are exact target symmetries. Additional particles and32move steps plus independent higher-particle restarts diagnose sensitivity. This is an approximation, not a guarantee of correct mode discovery. ESS alone is insufficient. A small frozen support is never the complete inference model and no log(particle-count) EIG claim is made. Independent reference EIG remains conditional on approximate beliefs; failed posterior stability blocks full training.

## Neural inputs and predictive outputs

Nine fixed physical reference probes cover the existing feasible box. For each candidate, the router receives its centered/scaled simulator response, nine centered/scaled anchor responses, and observed-history stage. It never receives theta coordinates, the hidden truth, or a source label. All responses are invariant to source exchange. A two-layer32-unit router produces four soft memberships per posterior configuration. Posterior-weighted memberships give alpha_e. Within each region, response mean, variance, skewness and occupancy, global response moments, candidate coordinates and stage condition a48/48tanh density head. Each expert outputs three Gaussian weights, locations and scales, not experiments. Variance includes the known observation noise floor. q_mix=sum_e alpha_e q_e is a normalized12-component continuous density, capable of multimodality. A one-head12-component control has its width chosen only to match parameter count, not EIG. Shared posterior datasets and equal update counts/simulation access are disclosed; runtimes are separately measured, not assumed equal.

## Information learning

J = E[-log q_mix(y|b,a)] +0.1 L_rank +0.02 L_response.

Training data include multiple counterfactual actions at every sampled posterior state; these actions need not be chosen by the planner. y is generated from posterior draws and fresh true Gaussian observation noise. Ranking references use an independent stratified posterior sample/noise stream and an exact weighted Gaussian predictive denominator for that approximate belief. They are entropy Monte Carlo estimates, not sPCE targets. L_rank averages softplus(-sign(I_i-I_j)(Ihat_i-Ihat_j)) only when |I_i-I_j| exceeds twice the combined reference SE. Near ties contribute no confident ordering. L_response=sum_e alpha_e Var_e[mu(theta,a)]/(Var[mu(theta,a)]+sigma^2) encourages physically distinguishable regions; it neither enforces equal occupancy nor rewards source-coordinate distance. Its small fixed weight is predeclared. There is no actor-critic, posterior variational loss, score matching or direct-action expert loss.

Ihat=H(q_mix)-H(y|theta,a), with differentiable24-node Gauss-Hermite integration over each mixture component. H(q_mix) is an approximate predictive entropy and is not an EIG bound. Held-out predictive NLL, PIT calibration, EIG bias/RMSE and uncertainty-masked ranking accuracy must pass numerical gates before expensive training. No between/within information decomposition is used or claimed.

## Training pseudocode

1. On seeds1004/1005 generate separate complete exploratory30-stage histories with the unchanged simulator; keep true theta outside all policy/model inputs.
2. Save posterior states at stages0,4,14,29; simulate8counterfactual actions and independent predictive/ranking observations.
3. Recompute held-out posterior states with independent4096-particle/rejuvenated chains; compare with2048-particle estimates.
4. Fit MoE and parameter-matched single density control for2000updates each, fixed Adam0.001 and gradient clip5, joint router/density objective above.
5. Gate predictive calibration, ranking, information error and posterior reliability before an additional12000MoEupdates. All pilot/control/development costs count against one14400second allowance; no clock reset or hyperparameter sweep.
6. Freeze last valid checkpoint. Do not choose a checkpoint or tune with seed1001. Only then perform one common2000/L1M reference evaluation if gates pass.

## Sequential selection pseudocode

At each stage, use current posterior to score24 feasible candidates and3 local refinements. Shortlist2; for each simulate2 possible observations, perform full Bayesian updates on forked complete beliefs and add the average best predicted second-step information. Execute only the highest combined score, observe once, update the actual belief and continue. At the last stage use immediate information only. Immediate versus two-step selection is an internal diagnostic with shared seeds/candidates. Two-step lookahead is bounded and not globally optimal over30stages. Actual online wall time includes all inference and hypothetical updates.

## Prior work and proposed scope

Predictive entropy selection and entropy subtraction are established: Houlsby et al., Bayesian Active Learning for Classification and Preference Learning, https://arxiv.org/abs/1112.5745 . Neural conditional mixtures are established: Bishop1994, https://research.aston.ac.uk/en/publications/mixture-density-networks/ . Amortized/variational information estimation is established: Foster et al.2019, https://papers.nips.cc/paper_files/paper/2019/hash/d55cbf210f175f4a37916eafe6c04f0d-Abstract.html . SMC design and rejuvenation are established: Drovandi and Pettitt, https://www.sciencedirect.com/science/article/pii/S0167947312002137 ; Gilks and Berzuini2001 resample-move is discussed in https://pmc.ncbi.nlm.nih.gov/articles/PMC2753529/ . This prototype proposes the particular response-space posterior router plus regional predictive heads, independent counterfactual ranking supervision and their bounded planning integration. This combination requires empirical and broader literature validation; no first-of-kind claim.

Unlike DAD it does not train a history-to-design policy with sPCE. Unlike native RL-sBOED it has no actor,Q-network,replay control learner or entropy-temperature updates. Unlike vsOED it has no variational theta posterior or actor-critic. Unlike ScoreBED it learns normalized predictive densities rather than marginal score fields. Resemblance to established predictive-MI surrogates and mixture-density models is explicitly acknowledged.

## Interpretation limits

Seed101 is not five-seed evidence. The established seed1001 bank is a previously consumed comparison bank, not untouched confirmation. A gate failure is a negative research result and prevents an official EIG claim. Equal wall allowance is not equal FLOPs; report algebraic simulator propagations, likelihood terms, optimization counts, measured CPU/GPU wall durations and missing hardware-active work separately. No claim of superiority, specialization, exact posterior or efficiency is justified before evidence.
