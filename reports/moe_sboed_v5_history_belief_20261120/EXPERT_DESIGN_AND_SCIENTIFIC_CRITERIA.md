# Policy-only history–belief integration:untrained V5 candidate

Scientific questions are fixed:
1. Does the four-expert MoE POLICY improve full-horizon EIG under fair comparison?
2. Does learned history/belief-dependent routing add EIG over independently trained uniform routing?
3. Are expert computations different AND conditionally useful to experimental decisions?

A declared role,feature difference,gradient norm or nonuniform weight is not evidence of useful specialization or routing EIG benefit. This source is implemented and CPU-audited but UNTRAINED;no performance result or success claim exists.

## Integration without new networks

Reuse the previous V5's ordered-history context64,expert-specific conditioners,commanders,256→128 expert projections,nonaffine LayerNorm,644→64→4 router,soft weighted sum128,128→32→2 head and bounded.2 weak branch. Parameter count remains371136,identical state keys and initial parameter/buffer values. Extra parameters0;extra neural networks0. No critic changes,no posterior-refresh mechanism,no auxiliary reward,no specialization labels,no teacher,no candidate-grid action selection.

Each existing conditioner still receives history context64,a14D belief-related view,remaining horizon1,and its existing4/6/6/4D command. Only the construction of the14D view changes by role:

|Expert|History–belief view|sBOED purpose|
|---|---|---|
|E1|Current approximate-posterior means,scales and dependence|Represent what remains uncertain,interpreted together with ordered history|
|E2|Last observed measurement's predictive innovation under the BEFORE-measurement approximate belief;predictive ambiguity,parameter/prediction correlations and resulting mean change|Represent what the last measurement distinguished or failed to explain|
|E3|Information supplied by previously observed designs,evaluated at CURRENT posterior component means;posterior-scaled unresolved-information matrix and current uncertainty scales|Represent coverage and remaining weak parameter directions|
|E4|Change between current and previous observed-prefix approximate-belief summaries|Represent recent information progress,conditioned separately on remaining horizon|

E2 uses the belief BEFORE the last measurement to compute its predictive innovation,not a future observation. E3's Fisher matrix is a feature descriptor,NOT a second posterior update. The posterior is never reweighted by already observed likelihoods,avoiding double counting. E4 uses an approximate provider's stage-to-stage changes;these are not exact posterior transitions. Existing expert semantic computations still receive their current belief/model information. The role view modifies the existing bounded command adjustment and per-channel gain,then existing projection/normalization. All four experts execute;the shared small head receives only routed features.

The numerical adapter here retains the original location-finding dimensions(117 history,4 parameters,8 mixture components,2D designs,T30,known noise). The scientific roles are Bayesian-design concepts,not spatial grid roles. SIR/power-grid adapters would require their own validated likelihood/Jacobian,history and dimension mappings. This is not plug-and-play cross-problem validation.

Current full belief is a sufficient decision state in an exact Markov Bayesian formulation;we do not claim history order must help when an exact sufficient belief is fixed. Here ordered history complements an approximate,compressed belief representation. The new information may be redundant or inaccurate;only EIG experiments can establish benefit.

## Controls and future validation

Native critic,posterior training,full-horizon variational actor–critic objective,prior,likelihood,physics,bounds and canonical evaluation remain fixed. The learned/uniform candidates must have identical actor architecture,initial common weights,feature evidence,expert training and budgets. Uniform is independently trained with exactly.25 routing throughout;router frozen/bypassed. No separate external NN is introduced.

Absolute EIG is primary:compare with previous V5,V4 and competitive native baselines using the original canonical bank/scorer(2000 trajectories,1M contrasts,T30). Disclose inherited pretraining,parameter counts and extra feature-computation cost. Identical parameter count is NOT identical compute. No canonical-bank tuning/checkpoint selection. One seed's case CI is not training-seed robustness;eventual five-seed paired uncertainty is needed for a reliable routing claim.

Structural MoE verification:four separate learned specialist computations,all active,normalized features,softmax gates,feature fusion,nonzero finite expert/router gradients,and measurable design sensitivity to weights. Scientific routing verification:learned beats independently trained uniform in paired EIG. Useful specialization:conditional contribution under small INTERIOR weight interventions on identical development histories and paired offline suffixes,not feature similarity alone. Single-expert inference ablations are diagnostic and may suffer distribution shift;individual experts need not solve the whole task or outperform collaborative MoE.

Executable audit passed:shape[B,4,14] role evidence;four128D expert outputs;128D sum fusion;unchanged initial actions/weights versus previous V5;exact uniform weights/router bypass;future-slot poisoning invariance;finite native-critic gradients;observations change role evidence at fixed stages;frozen deployment with autograd,optimizer construction and random-search traps. This is code correctness,not evidence of higher EIG.

No new GPU training was started. The already-running critic variant was not edited or interrupted and remains excluded from proposed-method development. This candidate is not promoted over previous V5. A formal pilot requires a separate locked execution step. Online action generation uses fixed neural/model computations only;no optimization,rollout search,MCMC or fitting.
