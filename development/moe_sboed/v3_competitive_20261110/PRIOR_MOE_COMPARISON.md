# Prior mechanisms and verified project evidence

## Primary literature

[Jacobs et al. 1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf): competing experts use a probabilistic mixture objective. Posterior responsibilities allocate learning according to how well a component explains an observed target; gate and component credit are coupled. Useful local competence is distinct from equal load. In BOED an action density explains a sampled action, not its information value. Reward-weighted policy credit is still needed; responsibility alone does not estimate competence.

[MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/): shared experts with task-specific learned gates model task relationships and differing feature requirements. It does not supply a conditional information-value signal for four complete design policies. Its task-specific prediction objectives are appropriate as declared preparation, not additional final rewards here.

[PLE](https://doi.org/10.1145/3383313.3412236) ([paper](https://datawhalechina.github.io/torch-rechub/file/PDF/2020%20%28Tencent%29%20%28Recsys%29%20%5BPLE%5D%20Progressive%20Layered%20.pdf)): explicitly separates shared and task-specific experts, then progressively routes representations across extraction layers to address negative transfer and competing task performance. Feature layers do not correspond to chronological experiments. Reusing this decomposition would be an adaptation, not a new stochastic BOED responsibility estimator.

[Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html): top-one token routing, capacity management, balancing and numerical techniques enable efficient sparse training. These address exposure and infrastructure. Capacity or uniform load is not evidence of conditional BOED competence; token dropping is inappropriate for required physical experiments.

[Expert Choice](https://arxiv.org/html/2202.09368v2): experts choose their high-affinity tokens within fixed capacity; tokens can receive different numbers of experts. This changes allocation and improves utilization, not conditional experimental information-value estimation. One online history cannot depend on a future batch of deployment histories.

[DeepSeekMoE](https://arxiv.org/html/2401.06066v1): fine-grained expert segmentation and shared-expert isolation aim to reduce redundancy, with load considerations and specialization ablations. Shared feature experts are not the same as an alternatively selected generalist action policy. A four-policy generalist/specialist design borrows an established principle but needs its own EIG competence evidence.

[DAD](https://proceedings.mlr.press/v139/foster21a.html): amortized adaptive policies and contrastive information bounds. Its training estimator is not adopted here. [vsOED](https://arxiv.org/html/2306.10430v1): variational posterior rewards, actor-critic policy optimization, and terminal/incremental information formulations are the inherited training foundation. Native approximate critics and replay occupancy do not become exact MI gradients merely by adding a mixture.

[RL-sBOED](https://proceedings.mlr.press/v162/blau22a.html): the sequential design problem becomes an MDP solved with reinforcement learning, including models lacking differentiable simulators. Routing via remaining return is therefore not a new BOED principle. [ScoreBED](https://arxiv.org/html/2607.08335v1): learns policy-independent score approximations and then optimizes policy gradients with them. This is a distinct source of approximation; copying its objective would violate the current task. None of these methods establishes novelty for the assessed v3 construction.

## Actual project source and results

Sources inspected under scratch_runs, with original bytes preserved:

|Method|Actual mechanism|Limit relevant to v3|
|---|---|---|
|Direct-head|Four logit heads mixed before feasible sigmoid action transform; shared encoder/native Q|Not a mixture of complete action densities; similar shared-target gradients possible|
|Top-1 competitive pilot 20261107|Categorical selection each observation, selected-expert path gradient, exact four-way gate Q expectation, matching categorical Bellman expectation|Already implemented the standard competitive native actor-critic core; candidate-control -0.024792, CI [-0.080816,+0.031232]|
|Option-MoE|Three-decision options, option-conditioned critic, four action experts plus frozen fallback|Temporal commitment and fallback are established options principles; not identical to per-step v3 routing|
|CIAS / information-specialized option|Joint-generated prefixes, complete resimulated remaining native returns, cross-fitted conditional-mean teacher, uncertainty-shrunk gate CE and counterfactual replay|Teacher/model drift, limited labels, and auxiliary gate updates; no useful specialization established; unsuccessful classification preserved|
|Collaborative Soft|Four feature slots and separately supervised capabilities, soft feature coordination, one design head|Feature tasks need not become useful design competencies|
|SG-MoE|Frozen competency anchors plus EIG-adaptive feature deltas and learned retention|Representation retention does not establish four alternative policy competencies|
|HEC-v1|Command-conditioned model queries, geometry/likelihood/Fisher/opportunity features, collaborative action head|All-four usefulness and benefit over stage-only commands unestablished|
|HEC-v2|Hypothesis-pair discrimination and predictive opportunity responses, anchors and joint native EIG|Task strengths E2/E3, only E3 corrected donor contribution; dynamic-stage difference +0.005817, CI [-0.045043,+0.056678]|

The earlier Top-1 and CIAS formulas used discounted native returns (documented gamma .9), whereas HEC-v2 uses gamma 1. They also used different parent lineages and policies. Their negative results do NOT prove a stochastic v3 with gamma 1 would fail. They do establish that categorical Q competition and remaining-return responsibility teachers have already been tried. No implementation failure has been established as the cause of their results; critic, teacher, occupancy and exposure explanations remain hypotheses.
