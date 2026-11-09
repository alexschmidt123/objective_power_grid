# Model, gradients and objective

Location likelihood: theta=(theta1,theta2) in R4, prior N(0,I4), two 2D sources; m(theta,a)=log(0.1+sum_j (1e-4+||theta_j-a||²)^(-1)), y~N(m,0.5²), a in [-4,4]².

For d_j=1e-4+||theta_j-a||² and S=0.1+sum_j d_j^(-1), ∇_{theta_j}m=-2(theta_j-a)/(S d_j²). Thus F(theta,a)=4∇m∇m^T. The local information matrix is its approximate-belief average. This is local sensitivity; expected pairwise normal KL=(m(theta,a)-m(theta',a))²/(2*0.5²) assesses nonlocal likelihood separation. Averaging iid parameter pairs gives Var_q[m]/0.5². Neither quantity replaces the policy objective.

For each diagonal GMM component use its eight sigma points mu±2 sigma_i e_i, each with component weight/8. This 64-node deterministic rule exactly matches the GMM mean and covariance but only approximates nonlinear integrals. At zero observations use the exact standard-normal prior moments. Native GMMs after t observations consume only the first t designs/observations. Variational posterior approximation and source-label ambiguities remain limitations.

Let u=tanh(Delta([S,concat_k alpha_k z_k])). The feasible design is a=a0+4u-a0|u| coordinatewise. For u≥0 this interpolates to +4; for u<0 to -4. u=0 gives a0 bitwise. The head starts at zero; command gradients become active after fusion begins learning. All expert and command gradients are checked on a disposable nonzero-head fixture.

Final objective is the inherited native Barber–Agakov surrogate J_q=E[log q(theta|h30)-log p(theta)]≤I(theta;Y1:30). Inherited normalized prior-mixture safeguards make q proper. Gamma remains 1 from SG, so native information increments telescope to terminal utility. Native actor update uses -E[Q_phi(h,pi(h))]; critic/posterior losses and replay/schedules remain native. Approximate off-policy actor–critic gradients are not exact mutual-information gradients.

The chain rule includes ∂Q/∂a × (∂a/∂z_k)(∂E_k/∂c_k)(∂C_k/∂eta), plus coordination and shared-history paths. Frozen model weights do not receive updates, but exact likelihood/Fisher/frozen-critic derivatives with respect to commanded designs remain available. No auxiliary reward is added in Phases B/C.

The proposed research mechanism is information-value-trained selection of heterogeneous model queries followed by joint feature integration around an inherited policy. Query networks, residual policies, soft routing and auxiliary preparation are established ideas. This composition is a hypothesis, not a proven novel method or improvement guarantee.
