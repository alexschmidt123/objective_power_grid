# V4 information decomposition

DEVELOPMENT_UNVALIDATED. C=f(theta) is a fixed prior-predictive partition frozen before policy training. For a realized observed history h_t, let Y=y_(t+1), Z=Y_(t+2:T) under the actual continuation policy. Because C is a deterministic function of theta:

I(theta;Y,Z|h)=I(C;Y|h)+I(theta;Y|C,h)+I(C;Z|h,Y)+I(theta;Z|C,h,Y).

Here I(theta;Y|C,h) integrates over C~p(C|h); future coarse information integrates over Y~p(Y|h,pi); future fine information integrates jointly over(C,Y)|h. The fixed continuation policy is included in every predictive distribution. Conditional information given random Y is an expectation, not a value evaluated at one unintegrated future observation. At the final stage Z is empty and both future terms are exactly zero. No four independent MI objectives are invented.

Finite particles: theta0 is the generating draw, theta1:L are iid prior contrasts. A fresh bank is shared across outer cases; it remains independent of each outer truth/noise. Let M=L+1, ell_j(k)=sum_(s<=k)log p(y_s|theta_j,d_s), logq_j=ell_j-logsumexp ell. Let n_c count the AUGMENTED prior bank including truth and pi_c=n_c/M. Then Jtotal(k)=logM+logq0(k), Jcoarse(k)=logsumexp_(j:Cj=C0)ell_j-logsumexp ell-log pi_C0, Jfine=Jtotal-Jcoarse. Using the same augmented bank makes Jtotal(0)=Jcoarse(0)=Jfine(0)=0 and avoids empty true groups.

E1=Jcoarse(t+1)-Jcoarse(t); E2=Jfine(t+1)-Jfine(t); E3=Jcoarse(T)-Jcoarse(t+1); E4=Jfine(T)-Jfine(t+1). Pointwise E1+E2+E3+E4=Jtotal(T)-Jtotal(t). Negative realized values are NEVER clipped. All reductions stay in log space. This algebraic identity is exact for the implemented densities; the densities are finite self-normalized MC proxies, not proven unbiased continuous-model MI estimators. Prior-group coverage, posterior ESS, ceiling saturation and independent contrast-sensitivity audits are retained.
