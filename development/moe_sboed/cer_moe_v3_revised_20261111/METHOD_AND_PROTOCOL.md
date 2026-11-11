# Revised CER-MoE V3: overlapping responsibilities

One candidate and one trained uniform-router control, in that order. Preserve
the validated CER complete-proposal policy and all old checkpoints. The four
slots are interchangeable next-design policies at all 30 stages. There are no
required clusters, exclusive domains, chronological roles or generalist slot.
No shared action head bypasses routing; physical coordinates are never averaged.

The native history encoder and Bayesian model are shared. Each independent
expert produces one entire bounded design. The native variational critic Q(h,a)
estimates remaining undiscounted information. Candidate probabilities are
(1-epsilon)softmax(Q(h,a_k)/tau)+epsilon/4. Tau is learned within [0.1,10].
One expert is sampled and its complete design executed. Uniform deployment and
each frozen individual are separately resimulated for evaluation.

Exposure is a policy schedule, not a reward: epsilon=1 for 200 updates, then
linearly decreases over 300 updates to 0.1. All four proposals are enumerated in
the actor expectation, so every expert receives a path gradient throughout.
The same epsilon is used in behavior probabilities, actor derivatives and target
Bellman expectations. The control always uses epsilon=1. Native Gaussian design
perturbations remain off-policy behavior exploration. Exposure prevents zero
responsibility, not competency collapse; report actual selection and gradients.

Holding the critic fixed during an actor update, J=sum_k pi_k Q(h,a_k) has both
sum_k d(pi_k)Q_k and sum_k pi_k grad_a Q_k d(a_k). Autodiff differentiates the
Q-dependent probabilities through proposals and the temperature. It does not
differentiate critic parameters. Categorical enumeration is exact for this
surrogate and includes the exposure mixture; no straight-through approximation
or detached routing score is substituted. Target values enumerate the target
actor's proposals and target critic under the same exposure schedule. The native
off-policy replay and approximate variational critic remain approximations to
pure full-horizon EIG; this is not an exact on-policy MI gradient theorem.

The preserved local audit is training data, not an evaluation bank. For each
history choose a probe using the first independent posterior/suffix replica.
Only histories whose chosen probe has a positive two-MCSE lower margin over HEC
in the second replica, and ESS >=32 in both replicas, can anchor differentiated
preparation. This is approximate MC evidence, not an exact-MI hypothesis test.
Choose up to four such observed history examples with a fixed random seed.
Soft Gaussian kernels on posterior mean/logvariance define overlapping sampling
weights over 8192 native replay prefixes, with a 0.25 uniform background floor.
Fewer than four examples are allowed; anchors can be reused. If none qualify,
preparation is uniform. No group occupancy or competence label is fabricated.
Each expert optimizes the original native action-conditioned Q on its sampled
history distribution for 200 steps at the inherited actor learning rate. The
shared encoder, Bayesian model and critic are frozen during preparation. A tiny
fixed independent final-layer perturbation breaks exact initialization symmetry;
it does not assign expert roles or add a diversity loss.

Both arms load the identical prepared weights. Critic/posterior/replay and native
schedules come from frozen HEC-v2 iteration16001. Actor Adam moments are fresh
at the inherited scheduled LR; new topology never inherits incompatible moments.
Each arm uses 2000 updates, 10000 replay episodes/batch, 1000 new trajectories
per update, 5 critic and 5 posterior updates per stage, gamma1, native optimizer
and target schedules. All budgets/settings are fixed before scientific updates.
No sweep, extra seed, training extension, extra reward or compute penalty.

Evaluation creates one NEW independent bank only after both arms freeze:
2000 trajectories, T30, 1M negative prior contrasts with positive included once.
Compare candidate, trained control, candidate with uniform routing, all four
candidate experts, frozen HEC-v2, native vsOED and compatible preserved baseline
policies. Full trajectories are resimulated under each changed policy. Primary
candidate differences use Holm7 and paired unadjusted95% t intervals. One trained
lineage cannot establish seed-robust superiority. Frozen external models differ
in historical training/provider budgets.

An independent 128-history panel checks critic ranking against posterior-
conditional resimulated suffix variational returns for all four proposed designs.
The same candidate suffix follows the intervention. Report MC uncertainty,
selection/value alignment, expert conditional strengths and stage-stratified
usage. Conditional estimates retain finite-particle/variational approximation
error. They are diagnostics, never new training rewards. No novelty gate stops
training. Negative, incomplete and failed outcomes remain explicitly labeled.
