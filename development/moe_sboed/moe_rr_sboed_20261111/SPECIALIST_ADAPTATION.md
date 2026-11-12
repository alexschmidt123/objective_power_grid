# Existing specialists, small adaptations and conditional evidence

Use the actual frozen HEC-v2 specialist packets, including preserved competency
channels and learned response features. Each new branch learns a small
history-conditioned transformation through the same remaining-information critic.
There is no independent competency label,expert role or forced diversity target.

Different pair computations send different action and probability-weighted
gradients to participating branches. The exact pair expectation allows learning
from every pair during training, while deployment executes only the sampled two.
A fixed10% uniform-pair mixture preserves nonzero exposure; it does not prove
specialization and cannot guarantee nonzero useful gradients after convergence.
Logs record each adaptation/router/head gradient and pair inclusion counts.

Shared preparation uses100 full native actor-critic updates with uniform distinct
pairs. It is performed once and the entire actor/target,critic/posterior,replay,
optimizer/scheduler and simulator RNG state is cloned into the two arms. Both
receive the same opportunity to learn nonzero specialized corrections. All base
HEC weights are frozen for this entire pilot,including preparation.

Frozen tests compare learned deployment with uniform Top2 deployment and remove
each new adaptation in complete resimulations. The base's original specialist
pathway remains during removal. A common-prefix panel removes one adaptation
at the next decision only,retains the actual joint latent draw solely for
simulation,and replaces all future noise/routing. It reports grouped paired EIG
effects with histories as statistical units and multiplicity correction.
Different probabilities or numerical correction values alone are not evidence
of intelligent routing. Distinct useful conditional contributions must be earned.
