# General sequential design processes

DEVELOPMENT_UNVALIDATED. The target is higher full-horizon EIG, not prescribed utilization.

With history h_t=(a_1,y_1,...,a_t,y_t), policy a_{t+1}=pi(h_t), joint likelihood is product p(y_t|theta,h_{t-1},a_t). The policy-independent prior and shared observation kernel define I(theta;Y_1:T|pi). Chain-rule conditional mutual informations sum exactly over time; the four proposed processes do NOT form such an additive identity.

E1 concerns remaining global spread. E2 concerns separation between plausible explanations. E3 concerns uncertainty in correlated parameter directions. E4 concerns precision within remaining explanations. Broad spread can arise within modes or between modes; poor identifiability can persist after apparent local concentration. All four processes can matter at any history, not fixed stages. Their roles are structural hypotheses tested through held-out full-horizon utility, not assigned physical parameter labels.

The domain-independent core contains no named physical parameters. Unknown dimension is a constructor argument. Separate weights are expected for each domain; there is no zero-shot transfer claim.
