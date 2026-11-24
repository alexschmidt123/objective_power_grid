# MoE-sBOED architecture review:primary literature before another pilot

Status:the new history–belief pilot is held before GPU preflight/training.Controller4139419 was suspended while waiting,with no children.No training process was interrupted.The previously running critic-uniform process4128057 was left alone.Source/checkpoints/results remain preserved.No new architecture is claimed validated by this review.

## 1.Primary references and what they actually support

**Jacobs,Jordan,Nowlan,Hinton(1991),Adaptive Mixtures of Local Experts.** Separate experts and a learned gate can learn allocation across cases.The paper studies supervised mixtures;it does not prescribe four scientific roles for Bayesian design or supply an EIG training recipe.Our transfer is the modular/gated architecture principle,not its loss or a guarantee of specialization.
Source:https://www2.bcs.rochester.edu/sites/robbie/jacobs.j.n.h.nc91.pdf

**Shazeer et al.(2017),Outrageously Large Neural Networks:The Sparsely-Gated Mixture-of-Experts Layer.** Section2,Eq1 defines a weighted sum of expert outputs.Section2.1,Eq2 explicitly describes nonsparse softmax gating.Their main experiments use sparse/noisy routing and auxiliary balancing.This supports our dense four-expert weighted FEATURE mixture as a legitimate operator;it does not justify importing sparse routing or forbidden balance losses.The paper uses comparable FFN expert architectures with independent parameters:identical layer shapes do not imply identical learned functions.
Source:https://www.cs.utoronto.ca/~hinton/absps/Outrageously.pdf

**Puigcerver,Riquelme,Mustafa,Houlsby(2024),From Sparse to Soft Mixtures of Experts.** Section2.1 Eqs1–2 use token-to-slot dispatch and output combination,with different softmax axes.Experts process weighted mixtures of input tokens.This named Soft MoE architecture is not merely four scalar weights multiplying already-computed expert features.Our current code should be called a dense soft-gated feature MoE,not a reproduction of this paper.Its vision results do not establish sBOED gains.A faithful token/slot implementation would require a separate explicit design and capacity/compute review;do not silently substitute it.
Source:https://arxiv.org/html/2308.00951v2

**Foster,Ivanova,Malik,Rainforth(2021),Deep Adaptive Design.** Section4.3,Theorem3 andEq17 exploit permutation invariance of observed design–outcome pairs under the paper's observation assumptions and use pooled pair encodings.History matters through information content;arbitrary storage order need not matter.This motivates set-consistent history encoding for the current static location-finding problem.It does not justify changing our native variational objective to DAD's contrastive training objective.DAD deploys an amortized forward policy,consistent with no online optimization.
Source:https://proceedings.mlr.press/v139/foster21a/foster21a.pdf

**Huan,Marzouk(2016),Sequential Bayesian optimal experimental design via approximate dynamic programming.** Sections2.2–2.3 formulate feedback/lookahead and posterior belief state,plus relevant physical state.Section4.1.1,Algorithm2 evaluates policies with a common assessment framework.Our inference is that common belief/physical state and remaining horizon should condition the entire MoE,and that full-horizon value is a global objective—not the private job of one expert.The numerical ADP solver is not proposed for deployment here.
Source:https://arxiv.org/pdf/1604.08320

**Shen,Dong,Huan(2025 journal paper;2023 preprint),Variational Sequential Optimal Experimental Design using Reinforcement Learning.** Section2.2 defines belief/physical state or full observed history and adaptive policy.Section3 develops native variational actor–critic policy learning.Different experimental utilities include parameter inference,model discrimination and goal prediction;they are not automatically separate experts in a single parameter-EIG task.We retain the existing native actor/critic/posterior training and do not add utilities or networks.
Sources:https://www.sciencedirect.com/science/article/pii/S0045782525003408 ;https://arxiv.org/pdf/2306.10430

## 2.Corrections to our current design reasoning

1.The V4 weighted-sum operator is not inherently an invalid or fake MoE.Routing usefulness remains an empirical question;current names/features do not establish it.
2.Soft-gated does not mean a faithful Soft MoE(2024) implementation.Use accurate terminology.
3.Posterior geometry,predictive ambiguity,sensitivity and sequential progress were our hypotheses—not four roles derived by these papers.They overlap and may provide redundant information.Geometry is also a representation property,not itself an optimal decision strategy.
4.All experts and the router need access to the relevant observed Bayesian state.No expert alone owns lookahead;all are optimized under the SAME full-horizon EIG objective.
5.For static location finding,latest-observation/recency emphasis and ordered temporal moments are not automatically justified by the likelihood symmetry.The held candidate's last-innovation/belief-change views should therefore not be treated as a literature-backed solution.This is a theoretical concern,not a confirmed implementation bug or demonstrated root cause of low EIG.
6.Order/physical state may genuinely matter in dynamic problems.Do not assume static exchangeability for SIR or power-grid dynamics.Verify the likelihood/state model and retain timestamps/physical state as needed.
7.A reported gain over an external baseline is a method comparison,not automatic causal attribution to the mixture architecture.Capacity,information providers,pretraining and compute differences must be disclosed.Controlled internal comparisons establish narrower claims.

## 3.Recommended reference architecture—not yet a new implementation

Use a conventional dense gated FFN feature mixture with four independently parameterized experts and a common Bayesian-state interface:

s_t = [history representation,current approximate belief,relevant physical state,remaining horizon]
z_k = E_k(s_t,v_k)
w = softmax(g(s_t))
z = sum_k w_k*z_k
a = bounded_small_head(z) + existing bounded weak correction

Here v_k is an optional role-specific known-model information view,not an optimal-expert label.The equations are our sBOED adaptation of the standard dense MoE operator—not a theorem that it improves EIG.The existing small feature head can remain;no unrestricted shared-history action bypass,no independent action averaging,no extra critic/inference/planning NN.Use native training unchanged.

Prefer comparable expert capacity and a common output interface.Independent parameters and conditional training can create specialization even when layer architectures match.Do not deliberately deny an expert basic state information to manufacture a routing gap.If constrained to current parameter budget,refactor existing policy modules rather than append new ones.Exact dimensions and checkpoint mapping must be audited before implementation;do not invent a paper-prescribed width.

For static location finding,base the history interface on pooled observed design–outcome representations,as justified by DAD's symmetry.Keep posterior/remaining-horizon inputs explicit.Do not add a new encoder beside the old one merely to increase capacity;reuse/replace existing policy encoding within budget.A theoretical invariant interface can still be imperfect if an order-sensitive approximate posterior is fed into it;that dependency needs explicit audit.

## 4.sBOED-grounded role hypotheses

These are proposed inductive biases,not claims taken verbatim from a paper or a decomposition of EIG:

|Expert bias|Bayesian-design rationale|Necessary caution|
|---|---|---|
|Global uncertainty allocation|Learn decisions when uncertainty is broad or dispersed|Large covariance alone does not imply an informative design|
|Nonlocal ambiguity resolution|Use predictive differences between plausible posterior explanations|Not a separate model-selection reward;include observation noise|
|Local identifiability/refinement|Use likelihood sensitivities and weak posterior-scaled directions|Fisher information is a local approximation,not exact nonlinear EIG|
|Information complementarity|Use observed information coverage and current uncertainty to evaluate what additional measurements may add|Repeated measurements can be optimal;do not impose a novelty reward or assume redundancy|

All four receive common observed history/belief and remaining horizon.All four learn the same native full-horizon EIG objective.All four remain active.The roles should guide inputs/computations,not force supervision,hard regime assignments or separate auxiliary objectives.If conditional contribution evidence does not support these roles,we must report that rather than rename features as specialization.

## 5.Evidence required before scientific claims

A.Real mixture:separate expert parameters,comparable feature outputs,normalized gates,gated fusion,finite gradients and controlled design sensitivity.These are implementation checks.
B.Routing value:learned versus independently trained uniform with identical actors/providers/budgets and common evaluation randomness.Report paired EIG intervals and eventually paired seed variability;weight variance alone is insufficient.
C.Useful expert differentiation:conditional contribution under small interior weight interventions at identical development histories,paired remaining-horizon effects,and inference ablations with distribution-shift limitations.Feature cosine differences do not prove utility.Standalone experts need not solve the complete task.
D.Absolute performance:original canonical bank/scorer and strongest eligible baselines,with native objectives and honest compute/pretraining accounting.Finite1M-contrast saturation remains a separate limitation.

No load-balancing/diversity rewards,no new online optimization,no critic augmentation,no posterior refresh,no automated architecture search,and no automatic additional seeds.The queued hand-designed pilot remains held pending a revised architecture specification based on this review.
