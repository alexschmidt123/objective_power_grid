## Bottlenecks: findings versus hypotheses

- **Confirmed:** all seven routed experts had gradient tensors at all 14,232 optimizer updates. Logged nonzero expert gradients argue against gross training starvation; tensor presence is not proof of competence.
- **Confirmed:** the shared feature norm averages 0.186664, versus 0.029860 for the routed sum. Shared-induced design changes are about 6.01 times larger. This is feature/action dominance, not a demonstrated information advantage of the shared branch.
- **Confirmed:** mean router entropy is 1.944691, close to log7=1.945910. Yet hard selections vary by stage and within stage; high entropy does not mean the router ignores history, and usage differences do not prove information specialization.
- **Empirical limit:** latest logged training sPCE 8.749205 is 0.261831 below the finite-L ceiling log8193=9.011035. Near-ceiling training can reduce information resolution; it does not establish that true EIG is optimal.
- **Empirical limit:** 100.0% of logged updates exceed the norm5 clipping threshold. Logged gradient median=69.2126, maximum=3075.7571. Temporal score-gradient variability includes parameter drift; it is not a fixed-policy variance estimate. Gradient noise as the cause of poor routing is plausible, not proven.
- **Unresolved:** feature redundancy in information acquisition versus a router failing to exploit complementary competence. The present Q intervals and excluded histories cannot discriminate these explanations.
- **Unresolved:** optimization convergence. The same development bank improves from10.30536 at5000 updates to10.61710 at10000; no same-bank final14232 development estimate was saved. Different final-bank scores are not a continuation of that curve.

The next justified modification is to improve and validate later-history posterior references and reduce variance in conditional remaining-information differences before assigning expert roles or changing expert count. Then test information-grounded credit assignment in a separately declared study. No new training, architecture change, checkpoint selection, or official reevaluation is performed by this report.
