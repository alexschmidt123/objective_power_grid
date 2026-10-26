# FH-MoE variance-reduction results

## Predeclared estimators and validity

Let Z=(generating theta, all observation noise) determine a complete rollout under a frozen direct policy, and let C be an iid L2000contrast bank. G(C;Z) is the exact pathwise terminal-sPCE adjoint. The simulator, likelihood and30-stage computational graph are unchanged. Contrasts do not affect histories, verified exactly.

For independent C_r, Gbar=(1/R)sum_rG(C_r;Z) has E[Gbar|Z]=E[G(C;Z)|Z]and Var(Gbar|Z)=Var(G|Z)/R. For the Gaussian prior,C and-Ceach have the same iid-bank marginal distribution. Therefore A(C;Z)=(G(C;Z)+G(-C;Z))/2has the same conditional expectation as G. Its variance is one half of single-bank variance plus one half of the covariance between its two members; sign reflection does not guarantee negative covariance for this nonlinear likelihood. Equal-work comparison uses independent pairs(G(C)+G(D))/2. This is a direct symmetry argument, not learned antithetic sampling. See[Ren,Zhao,Ermon2019](https://proceedings.mlr.press/v97/ren19b.html)for broader unbiased antithetic methodology.

Common random numbers share Z and base C across estimator comparisons, preserving marginals and supporting paired uncertainty. This coupling can reduce difference variance; it does not by itself reduce a single estimator variance. No posterior/importance sampler or control variate was used. Sampling prior theta after fixing an observed prefix would be invalid: p(theta|h) is proportional to p(theta)times the prefix likelihood. The generating theta is not substituted for that posterior.

Crucially,E[normalize(Gbar)]need not equalE[normalize(G)]. Raw averaging is unbiased for the SAME finite-Lexpected raw gradient, while normalized directional labels change. Passing a directional gate after averaging is therefore not an unbiased estimate of the old unit-label functional, nor evidence of expert competence. L8000changes the finite-contrast objective itself and is only a sensitivity study, not a guaranteed true-EIG bias correction.

## Independent paired context comparison

Three fresh splits of8000complete histories each; disjoint fitting/independent fitting/test seeds98261041/42/43with fixed per-batch offsets. L2000,float64. Four independent base banks and four independent partners plus four reflected banks per trajectory. Existing gates and ridge settings unchanged. Before/after rows below use the SAME fresh histories,Nand independent splits; they are not comparisons between different-sized samples.

|Stage|Single-bank min coherence|Four-IID min coherence|Single-bank min SNR|Four-IID min SNR|Single/Four-IID kappa|Four-IID gate|
|---|---:|---:|---:|---:|---|---|
|1|0.02581|0.02120|0.7810|0.7387|0.0346/0.2113|FAIL|
|10|0.04660|0.05315|2.1316|2.5404|0.5680/0.6645|FAIL|
|20|0.04355|0.04279|1.8842|1.6947|0.6545/0.7761|FAIL|
|29|0.08079|0.10629|2.6919|3.7244|0.7642/0.8176|PASS|

Only stage29passes all point-estimate gates with four independent banks; eight independent banks also qualify only stage29. The predeclared requirement was at least TWO primary stages among10,20,29. It fails. The stage1negative control never passes the complete gate, even when assignment kappa alone increases.

Stage29weak-context coherence bootstrap95%intervals for its two split halves are[.0773,.1440]and[.0805,.1499]; SNR intervals[2.70,5.07]and[2.78,5.21]. These span the acceptance boundaries. The point pass is exploratory, not independent confirmation. Stage10and20mean coherence remains below.1after either4or8IIDbanks. All fixed-label bootstrap diagnostics and original MSE intervals are preserved in variance_results.json.

## Antithetic versus equal-cost independent pairs

|Stage|Trace-variance ratio,antithetic/IID pair|Paired trajectory-bootstrap95%CI|
|---|---:|---|
|1|1.0303|[0.9083,1.1681]|
|10|1.0730|[0.8923,1.3246]|
|20|0.9671|[0.8524,1.1482]|
|29|1.0132|[0.7820,1.2985]|

No interval has an upper endpoint below1: the predeclared extra antithetic-benefit criterion fails at every stage. The eight-evaluation antithetic mean does not qualify any stage. Raw conditional expectation is preserved mathematically; finite-sample componentwise paired mean-difference intervals are reported, not used as proof of zero true-EIG bias.

## Larger contrasts and bias limits

The selected512paired cases use four repeats each of nested L2000andL8000banks. L8000gradient trace variance was higher in all stages; bootstrap95%ratio intervals were1.26–2.88,1.34–4.15,1.14–6.80and1.53–9.20. Mean paired direction dots between four-repeat averages ranged.61–.72. This is finite-bound/estimator sensitivity, not evidence that more contrasts generally worsen final EIG. Neither an infinite-contrast gradient nor exact-EIG bias was computed.

## Compute and preservation

New UbuntuPC GPU-locked task wall:35.100006s;predeclared maximum1800s,internal soft limit1650s with60sreserve and hard1800salarm.636gradient calls,299264evaluated complete trajectories,18333457920scalar conditional likelihood/parameter-stage evaluations,peak2853622784bytes. All operations are in GPU_ledger.json. Zero optimizer updates and no official benchmark evaluation. No excluded GPU pilot or GPU retry occurred. CPU tests/analysis,source inspection,rendering,transfer and LabPCvalidation are additional; do not call GPUwall a complete end-to-end project cost or measured GPU-kernel busy time.

CPU timing records:Ubuntu precollection tests1.96s,postanalysis tests1.97s;saved analyses.5127/.5030s;initial new analysis1.1580s;expanded fixed-label uncertainty analysis2.5785s. CPU process and wall can overlap GPU wall; do not sum them as exclusive computational expenditure. LabPC tests/audit timing is recorded at integration. Untimed inspection/transfer/figure overhead prevents an exact entire-workflow total.

Raw tensors are SHA256manifested and preserved in dated isolated campaigns,excluded from Git. Summaries,plan,collector source,collection-time analysis,figures/CSVand logs are committed from LabPC only. Six v1foundation files remain byte-identical. Numerical figures are generated by tools/report_gradient_variance.py.

The future benchmark protocol remains LabPCseed10114400charged seconds including all trainable stages; Ubuntuseed102matches its completed optimizer count. No training was authorized or launched by this audit, and matching updates would not eliminate hardware effects.
