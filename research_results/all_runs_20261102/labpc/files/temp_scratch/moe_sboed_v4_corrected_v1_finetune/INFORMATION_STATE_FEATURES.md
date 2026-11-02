# Offline predictive uncertainty candidate
Feature U(h)=sum_d Var_approx[f(theta,d)|h] / sum_d Var_prior[f(theta,d)], at a frozen3x3 reference grid with coordinates -2,0,2. This is posterior predictive dispersion, not a full uncertainty or identifiability metric. Prior particles exclude generating truth; only previous observed actions/observations determine normalized likelihood weights. The grid is an offline domain-adapter choice, absent from deployment.
Threshold .75; exclusion margin .1;ESS>=64;max weight<=.1;independent-half classification agreement and dispersion difference<=.2. Independent reference bank131072particles;referenceESS>=512 andmaxweight<=.02. No true theta as posterior representative.
128 held-out histories;89 eligible;cohort counts[0,89];classification agreement 0.988764;90th-percentile absolute ratio error 0.036686. The numerical feature is reasonably reproducible on eligible cases, but two-cohort coverage FAILS. No credible distinct information-state cohort gradient can be estimated under this frozen rule.
Only stage1 was a candidate for admission; later stages were not validated and cannot justify new corrections. This does not test competing explanations or poorly identified combinations. Independent finite-prior reference is not an exact posterior or an MCMC convergence guarantee.


Status: BLOCKED_INFORMATION_COHORT_VALIDATION. No fine-tuning or official evaluation was launched. No STABLE changes, GitHub push or changes to previous V4/Ubuntu SIR.
