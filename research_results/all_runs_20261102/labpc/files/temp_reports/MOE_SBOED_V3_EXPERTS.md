# Four process experts and learning signals

For an amortized K=4 Gaussian mixture q(z|h)=sum_k w_k N(mu_k,L_k L_k^T) in benchmark-transformed parameter coordinates, define m=sum w mu; W=sum w C_k; B=sum w(mu_k-m)(mu_k-m)^T; C=W+B. These identities describe the learned mixture, not a certified physical posterior.

Exactly four independent 16-unit SiLU expert networks receive the shared history vector and different operators:

- E1: m and a bounded total-covariance operator log(1+diag C) on the diagonal plus C/(1+abs C). Intended capability: global spread-sensitive representation.
- E2: m and B/(1+trace C). Intended capability: contrasts between mixture-component explanations. A mixture component is not automatically a posterior mode, and this is not physical predictive disagreement.
- E3: m and R=diag(C)^(-1/2) C diag(C)^(-1/2). Intended capability: correlated poorly resolved directions, in the declared parameterization.
- E4: m and W^(-1)/(1+trace W^(-1)), computed by Cholesky solve. Intended capability: within-explanation precision. Add 1e-5 I for numerical stability.

Matrix entries are tanh-bounded before expert inputs. These operators are invariant to mixture label permutation, but not arbitrary reparameterizations of unknowns. No expert emits an independent experiment proposal. Independent networks, distinct structured operators, and joint nonlinear fusion provide a concrete mechanism hypothesis; useful specialization still requires measured conditional utility.

All four optimize terminal sPCE through their selected paths. ONE auxiliary proper conditional cross-entropy supports the common uncertainty representation. There are no four fabricated EIG losses, no strategy labels, and no forced expert utilization.
