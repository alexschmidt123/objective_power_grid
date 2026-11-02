# IR-MoE-v2 rejection diagnosis

All198 saved queries were audited. Exactly76 stopped on low posterior ESS;122 completed conditional comparisons and none exceeded the2.58-SE threshold. The maximum complete-case SNR was2.511224.55 completed comparisons also disagreed in sign between posterior replicates. Sign disagreement is an additional failure, not an extra disjoint count.

| Stage | Probes | Low ESS | Completed | Median abs effect | Median MC SE | Maximum SNR | Replicate sign disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|
|1|50|0|50|0.00143430|0.00167892|2.511224|17|
|5|50|1|49|0.00125480|0.00205281|2.413913|23|
|15|49|28|21|0.00127829|0.00264335|2.478651|13|
|25|49|47|2|0.00000360|0.00001141|0.500529|2|

No evidence supports simply lowering the threshold. Early completed estimates are mostly small or noisy. Middle/late importance degeneracy is independently documented. Near-zero point estimates do not prove negligible true effects. V2 had no across198-probe multiplicity correction; that omission could increase false labels, not explain why none passed. The v2 pooled fantasy SE conditioned on finite posterior references and did not fully measure posterior approximation bias.

No demonstrated v2 indexing, PL log-probability, sPCE formula or hybrid-gradient bug was found in this audit. Exact enumerable tests validate coefficient1 and the earlier-history routing derivative. A weighted-reference comparison bug in NEW v3 development was fixed before the pilot and regression-tested; it is not attributed to v2.
