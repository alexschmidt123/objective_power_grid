# Stochastic Top2 and the native policy gradient

Four transformed logits ell_k=10tanh(raw_k/10) define weights exp(ell_k).
Sampling without replacement gives ordered probability
P(i,j)=exp(ell_i)/sum_l exp(ell_l) * exp(ell_j)/sum_{l≠i} exp(ell_l).
The correction is symmetric in its selected feature set, so each of six unordered
pairs has P({i,j})=P(i,j)+P(j,i). Stable logsumexp/logaddexp calculations evaluate
this identity exactly up to floating-point roundoff. Main learned routing uses
q_ij=0.9P({i,j})+0.1/6. Preparation and the matched control use q_ij=1/6.
Sampling one of these six categories always selects two distinct experts.
The exploration mixture is included in all behavior,actor and target quantities.

For the native critic Q_psi(h,a) and each complete pair-corrected action a_ij,
J(h)=sum_{i<j} q_ij(h) Q_psi(h,a_ij(h)). Holding psi fixed during actor improvement,
its exact derivative is
sum (d q_ij) Q_ij + sum q_ij grad_a Q_ij d a_ij.
The first term is the categorical score-function expectation, equivalently
sum q_ij Q_ij d log q_ij. The second differentiates the selected feature correction.
We enumerate all six terms; no gradient is taken through sampled hard indices.
No auxiliary routing reward,entropy bonus,load-balance or diversity term exists.
The critic target also enumerates six target-policy actions and probabilities.

This is the exact derivative of the finite-pair native replay actor surrogate.
The learned critic,off-policy replay and variational posterior remain native
approximations; the method does not claim an unbiased exact on-policy MI gradient.
Gamma1 and the unchanged normalized native posterior reward retain the
full-horizon variational information objective. Reward increments telescope to
log q_T(theta|history)−log p(theta); validation checks the identity on the complete
native replay batch. sPCE is used solely for frozen EIG evaluation.

At initialization W_out=0 makes all pair actions identical. Router derivatives
sum to zero, and adaptation gradients are blocked by W_out. Head gradients can
already be nonzero. A shared100-update uniform-Top2 native EIG preparation opens
the correction pathway before any learned-routing preference is expected.
Tests check the zero block,nonzero-head gradients,finite differences,independent
score-function identity,sampling frequencies and actual preparation reload.
