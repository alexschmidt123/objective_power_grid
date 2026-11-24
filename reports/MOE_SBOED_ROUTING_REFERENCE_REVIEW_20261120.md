# Routing design and RL:primary-paper clarification

No new architecture or router algorithm is implemented by this review.The new pilot remains held.The four hand-assigned expert roles are NOT an accepted scientifically established specialization scheme.

## What papers actually train

1.Shazeer et al.(2017),Section2.1 Eqs2–5:linear softmax scores,with noisy top-k for sparse routing.Selected weights/gate parameters train by backpropagation with the task model;this is not a separate REINFORCE expert-selection agent.Their explicit nonsparse softmax is a valid dense alternative.Reference:https://www.cs.utoronto.ca/~hinton/absps/Outrageously.pdf

2.Puigcerver et al.(2024),Section2.1:token/slot logits produce dispatch and combine distributions by softmax over different axes.The differentiable layer learns through the task loss;no separate routing RL is required.This named Soft MoE mechanism differs from our four-vector weighted feature sum.Reference:https://arxiv.org/html/2308.00951v2

3.Bengio,Bacon,Pineau,Precup(2015/ICLR2016 workshop):conditional block activation is a stochastic computation policy trained with policy gradients.The objective considers predictive quality/computational sparsity and includes regularization.This is an example of RL routing/conditional computation,not a native sBOED recipe,and its compute/diversification terms conflict with our no-extra-reward rules.Reference:https://arxiv.org/abs/1511.06297

4.Kim,Kang(EMNLP Findings2025),Exploration-Driven Reinforcement Learning for Expert Routing Improvement in Mixture-of-Experts Language Models:retain the existing sparse router,compare original versus perturbed routing paths,and use their prediction-performance difference as a PPO-style advantage.The method does not require another reward model.Its optional knowledge-overlap training and sparse-selection setting are additional algorithmic choices;not a drop-in recipe for all-active soft feature routing or native EIG.Reference:https://aclanthology.org/2025.findings-emnlp.1282.pdf

5.Dikkala et al.(EMNLP2023),On the Benefits of Learning to Route:studies softmax routers trained with gradient descent,including cluster-structured settings where learned allocation emerges.The theoretical conditions/experiments do not establish a theorem that any sBOED router must beat uniform;some studied structures do not show the same routing advantage.This supports testing state-conditioned specialization rather than inventing four semantic labels.Reference:https://aclanthology.org/2023.emnlp-main.583.pdf

## Our native RL already trains the soft gate

Let s_t be the observed Bayesian design state,including history/current approximate belief,relevant physical state and remaining horizon.Let z_k=E_k(s_t),w=softmax(g_phi(s_t)),z=sum_k w_k z_k,and a=D(z,s_t),where D includes the small action head,bounds and fixed weak correction.

For ONE native actor-loss sample,l=-Q_omega(h_t,D(z,s_t)),the gate-logit derivative,holding expert features/state fixed,is exactly:

    d l / d ell_k = -w_k (z_k-z)^T grad_z Q_omega(h_t,D(z,s_t)).

This is chain-rule credit assignment inside the native deterministic actor–critic update.It is not a new reward,a new critic,an extra EIG estimator or a separate categorical RL policy.The Jacobian through the action head and action bounds is included in grad_z Q.The native critic is intended to represent remaining-horizon variational information return;this derivative alone does not prove accurate EIG credit.

The expression identifies the required conditions:experts must differ in action-relevant directions,the action head must transmit those differences,and the native value gradient must assign useful credit.Similar features,decoder insensitivity or inaccurate gradients can all make routing ineffective.A larger softmax network does not resolve those issues automatically.

At deployment,Q is absent and no routing/action optimization occurs.Only the fixed actor forward pass produces gates/features/design.No posterior fitting,MCMC,rollout search or weight updates.

## Why the expert specification must change too

A specialist should be defined by CONDITIONAL USEFUL CONTRIBUTION to design decisions over Bayesian states,not a name such as geometry/progress and not merely different weights.Same FFN architecture with independent parameters is standard;useful specialization must still be demonstrated empirically.All experts should retain the common relevant state,while the router learns context-dependent allocation.No requirement that each expert independently solves the entire task.

The appropriate specialization domain for sBOED is belief/design state—not a partition of sensor locations or token capacity.The common full-horizon information objective is unchanged.Posterior uncertainty,ambiguity,identifiability and horizon may be useful POSTHOC descriptors of learned regimes,but do not create optimal-expert labels or new supervision by themselves.No fixed four-role decomposition is justified by these papers.

A separate ERL/PPO router would change the training algorithm.Exploration of routing alternatives can first diagnose credit assignment OFFLINE,but must not silently become a new objective,extra episodes for learned routing or online search.Under current constraints,the defensible first reference is a standard differentiable gate trained inside the existing native actor objective,with explicit state/input/gradient audits and matched uniform comparison—not another auxiliary network.

The open scientific test remains:does learned allocation among conditionally useful experts produce higher full-horizon EIG at matched information/budgets?No architectural citation or gradient equation guarantees that result.
