# Candidate non-chronological history conditions

FROZEN PREFLIGHT COMPLETE; design competence and full 30-stage support remain INCOMPLETE.

Let q_h be the inherited SG amortized GMM used by frozen HEC-v2, approximated by its existing 64 weighted quadrature nodes. This is not a certified conditional posterior. Let C_h be its covariance and a_ref(h) the frozen HEC-v2 action.

Coverage score: c1(h)=tr(C_h). Broad coordinate covariance can conflate exchangeable-source label ambiguity with spatial uncertainty. It is a candidate diagnostic, not a validated explanation of where a coverage policy is superior.

Discrimination score: c2(h)=E_{theta,theta'~q_h} KL[N(m(theta,a_ref),.25)||N(m(theta',a_ref),.25)]=4 Var_{q_h}m(theta,a_ref). This uses the exact Gaussian likelihood and approximate-belief quadrature. A high score means hypotheses are distinguishable at the reference action, not that the reference action is poor or a separate expert improves remaining EIG. It is not a direct multimodality test.

Weak-identifiability score: c3(h)=-log(1+lambda_min(D_h F_hist D_h)), where D_h=diag(sqrt(diag C_h)) and F_hist=4 sum_{i<t} E_{q_h}[J(theta,xi_i)J(theta,xi_i)^T]. This is a current approximate-posterior-averaged accumulated Fisher diagnostic with diagonal scaling; it is not the posterior precision or exact observed information. J is independently autograd-checked. Weak directions can reflect label symmetries, numerical scale, or genuine ambiguity; no policy benefit follows automatically.

Predeclared preflight scores use calibration upper-third thresholds at each sampled stage, then s_k=.1+1[c_k>threshold_t], s0=1, normalized for d_k proportional to s_k d. Groups may overlap. The positive floor keeps every expert eligible; this is a diagnostic preparation weighting, not an equal-utilization objective. Strict inequality gives no arbitrary split when the information state is identical at t=0.

128 calibration and 128 held-out episodes per stage; all 1280 replay episode indices are distinct. No theta field or future observations are used. Prefix mutation checks confirm future saved entries have no influence. Thresholds were not selected from expert wins or EIG scores. The parent replay distribution is mixed historical training occupancy, not fresh v3 occupancy. No training distribution or threshold is finally locked.

Held-out upper-group prevalence:

|Stage|Coverage group|Discrimination group|Weak-identifiability group|
|---|---:|---:|---:|
|0|0.0%|0.0%|0.0%|
|3|33.6%|29.7%|30.5%|
|12|31.2%|22.7%|35.9%|
|24|27.3%|40.6%|27.3%|
|29|24.2%|34.4%|38.3%|

At t=0 the information histories are identical; all three scores tie, and no condition-specific preference can be inferred. All experts could still be sampled there. Later groups vary within stage, with pairwise Jaccard overlaps and weighted effective sample sizes in evidence/FROZEN_HISTORY_AUDIT.json. This validates distinct measurable conditions only. It does not establish useful design competencies, reliable posterior calibration, or conditional comparative advantages. Additional stages must be audited before any training lock.
