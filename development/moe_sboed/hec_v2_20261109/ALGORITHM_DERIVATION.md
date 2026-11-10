# Query-response anchored collaboration

Location likelihood is y~N(m(theta,a),0.5²), m=log(0.1+sum_j(1e-4+||theta_j-a||²)^(-1)), theta~N(0,I4), a in [-4,4]², T30. The native normalized variational information objective and optimizer remain unchanged, gamma1. There is no sPCE training or additional final policy reward.

For opposed quadrature weights p+ and p-, cross-hypothesis expected Gaussian KL at a is [Var+(m)+Var-(m)+(E+m-E-m)²]/(2 sigma²). This explicitly captures nonlocal discrimination. Sensitivity remains local: ∇theta_j m=-2(theta_j-a)/(S d_j²), F=∇m∇m^T/sigma², d_j=1e-4+distance² and S=0.1+sum1/d_j.

For an empirical eight-point belief with weights1/8, hypothetical observation y at design a gives w'_i ∝ exp(-(y-m_i(a))²/(2sigma²)). Future likelihood separation is evaluated under w' at feasible follow-up probes. Three predictive nodes match predictive mean/variance but only approximate integration. Future coordinates vanish at t29. These operations are differentiable in the commands; fixed systematic support selection is based on detached frozen belief weights. They introduce no actual future observation into the policy.

Let Lk be live v1 features and Ak fixed E1/E3 functions. First-slot features are (A1,L2,A3,L4). Response slots are (L1-A1, R2, L3-A3, R4), where R2/R4 concatenate 12 exact semantic coordinates and 52 learned coordinates. Fusion is P=W0[s;concat(gk*firstk)]+V concat(gk*responsek), followed by the inherited ReLU/output layer and bounded residual map a=a0+4tanh(d)-a0|tanh(d)|.

Initialize V's E1/E3 columns from W0's corresponding columns and its E2/E4 columns to zero. The inherited specialist weights remain unchanged. The candidate's initial action matches HEC-v1 exactly: E1/E3 differences are zero and new E2/E4 contributions are initially zero. E1/E3 gradients remain available through V, avoiding a zero-times-zero starvation trap. New E2/E4 response maps can learn from their nonzero prepared features immediately; adapters receive EIG credit as those maps learn.

The coordinator is inherited G([s,L1:4]) plus a zero-initialized response gate. Pure native ∂Q/∂a gradients propagate through fusion, query-dependent responses, likelihood/Fisher/predictive updates and commands. This is a new policy mechanism built with the existing actor–critic estimator, not a newly invented information-gradient estimator. Frozen weights are buffers excluded from optimizer/Polyak loops; all compatible inherited Adam objects are retained.

The exact joint information objective is approximated by the same native variational bound Jq=E[logq(theta|h30)-logp(theta)]. Proper q and gamma1 make incremental rewards telescope. Finite off-policy actor–critic training has no monotonic-EIG guarantee. Preparation's second-step/tail labels are auxiliary supervised targets only in A, not new rewards in B/C.
