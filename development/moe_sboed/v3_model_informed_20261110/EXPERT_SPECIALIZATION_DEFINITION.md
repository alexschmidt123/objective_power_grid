# Operational specialist preparation

4800 distinct parent replay episodes:128 training and32 heldout prefixes per stage for all30 stages. Only history-prefix data enter the policy. Generalist weight1;specialist weight .1+1[score exceeds stage-specific training upper-third quantile],normalized by training mean. Conditions overlap and do not split tied prior histories arbitrarily.

Scores:posterior covariance trace;expected pairwise Gaussian likelihood KL at the reference design;and -log(1+lambda_min(D F_past D)),where D is diagonal posterior standard deviation and F_past=4 sum_{i<t} E_q[J(theta,xi_i)J(theta,xi_i)^T]. These are approximate-belief diagnostics,not known expert-optimal regimes. Label symmetries and posterior miscalibration can confound them.

Prepare action heads only for200 Adam steps,lr.001,batch256,using weighted sampled-design Q from the frozen trained parent native critic. No new simulator trajectories. Encoder,router,critic,posterior and reference stay fixed. This is a remaining-value design surrogate,not observed EIG or a reconstruction objective. Preparation is physically shared once and charged equally to both arms.

Afterward the preparation optimizer is discarded. New expert/router main Adam moments are fresh;compatible encoder Adam moments are inherited. Main native EIG updates have NO curriculum multiplier or auxiliary. Roles need not persist or prove useful;heldout intervention returns must establish actual competence. All experts remain interchangeable and available at every t.
