# Full-horizon native variational mixture gradient

For frozen native posterior weights w and zero extra physical rewards in location finding, r_t^w=log q_{w,t}(theta|h_{t+1})-log q_{w,t-1}(theta|h_t), q_{w,-1}=prior. Gamma=1 makes the sum telescope to the terminal native variational information score. Its population expectation is a variational information bound, not exact MI. Native q training and critic approximation remain separate approximations; replacing them by sPCE is excluded.

Define J_w(pi)=E_pi sum_{t=0}^{29}r_t^w, with the native terminal convention retained. For a legitimate stochastic mixture and on-policy histories,

    grad J_w = E sum_t A_t [grad log g_{K_t}(h_t)+grad log pi_{K_t}(a_t|h_t)].

A_t can use an action-independent history baseline. A_t=Q_t^pi(h_t,a_t)-V_t^pi(h_t) is the remaining variational return, not a greedy information score. Replacing Q by a learned critic introduces critic bias. Replay histories from another policy add an occupancy approximation; action importance ratios alone do not correct history occupancy.

Conditional posterior mixture responsibility r_k gives

    grad log pi(a|h)=sum_k r_k(h,a)[grad log g_k(h)+grad log pi_k(a|h)].

The fixture enumerates all 64 joint trajectories of a non-myopic two-decision toy problem with four experts and two actions, and checks the full time-summed score gradient against exact differentiation. This is an independent algebra fixture, not a full location-finding training integration.

Bounded continuous policies can use z~N(mu_k(h),diag sigma_k(h)^2), a=4 tanh z. For interior actions,

    log pi_k(a|h)=log Normal(atanh(a/4);mu_k,sigma_k)
                 -sum_i[log4+log(1-tanh(z_i)^2)].

Stable log Jacobian is log4+2(log2-z-softplus(-2z)). No clipping density is substituted. Boundary atoms introduced by the historical Gaussian-noise-plus-clipping collection policy are not represented by this continuous density. Actor, data collection, stochastic Bellman target and saved RNG all need coordinated changes; deterministic native actor code cannot simply receive a log-probability term.

A compatible fixed-history critic surrogate is F=E_{h~replay}sum_k g_k E_epsilon Q(h,4tanh(mu_k+sigma_k epsilon)). Expert path derivatives and a categorical score or exact finite gate expectation differentiate F. Target values must integrate expert AND continuous-action randomness. A stored action plus log-probability from an old behavior policy does not by itself validate an on-policy score gradient. This is mathematically feasible and is not an assertion of implementation impossibility.

For G_k(h) the appropriate native gradient contribution, resampling q_k(h)=s_k(h)d(h)/Z_k with d/q_k yields the same E_d G_k. Leaving the correction out yields E_d[s_k G_k]/Z_k, generally a different vector. A positive score is not enough to guarantee it is the original gradient. Parameter-only positive preconditioning can be an optimizer for the same objective, but history-dependent selection weights are not automatically such a preconditioner. The toy counterexample explicitly detects the gradient change. Correction does not preclude variance reduction; proving or measuring a useful estimator is a possible research contribution, not accomplished here.

Native off-policy deterministic training already has a documented surrogate. A new biased responsibility update could be studied honestly, but its relation to the original objective and difference from CIAS would have to be specified before training. No universal unbiased-MI claim is made.

Actual fixture receipt:

```json
{
  "status": "PASS_COMPONENT_FIXTURES_ONLY",
  "seconds": 0.21248311316594481,
  "optimizer_updates": 0,
  "simulator_calls": 0,
  "exact_two_step_trajectories": 64,
  "score_gradient_max_error": 7.632783294297951e-17,
  "marginal_responsibility_gradient_max_error": 0.0,
  "bounded_density_mass": 0.9999999999999937,
  "unjustified_specialist_weight_gradient_change": 0.3014194735342817,
  "importance_corrected_history_gradient_matches": true,
  "exploration_distribution_derivative_correct": true,
  "variational_telescope_at_gamma1": true,
  "full_native_v3_integration_test": "NOT_IMPLEMENTED_NOT_PASSED",
  "no_v3_checkpoint_created": true
}
```
