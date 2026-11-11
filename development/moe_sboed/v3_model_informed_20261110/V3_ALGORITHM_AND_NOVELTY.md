# Model-informed complete-policy competition

One frozen HEC-v2 reference/belief provider supplies a reference action and the inherited approximate SG GMM quadrature. Four independent64-hidden-unit action heads share a trainable128-dimensional history encoder. Head0 is a generalist;heads1-3 receive coverage,discrimination and identifiability preparation. Each emits a complete next-design distribution at every t;none is an ordered experimental stage.

b(h)=atanh(clip(a_ref(h)/4,-.999,.999));mu_k=b+.5tanh(E_k(h));z~N(mu_k,.025^2 I);a=4tanh(z). The reference is shared computation,not a fifth selected expert or an executed prior experiment. A bounded residual class protects a strong parent but may limit specialization. Initial stochastic V3 differs from deterministic HEC-v2;initial-policy EIG is reported separately.

At each mean proposal compute9 model features:predictive mean,expected pairwise Gaussian KL,four diagonals of posterior-scaled expected Fisher,logdet(I+F_scaled),and proposal coordinates. The router uses stage,history,belief mean/variance,and these proposal-dependent features. g=.95softmax(router)+.05/4 is used in collection,targets,gradients AND deployment. Sample K,then a from that expert,simulate y,and reroute. No physical action averaging.

Model-feature gradients through the actual proposals remain active. Selected-action credit reaches the selected expert;allocation derivatives can also reach other heads through their own proposal features. This is the composite policy gradient,not a separate reward. The approximate belief and quadrature are not exact posterior inference. All four dense proposals are computed;no sparse-compute claim. This composition may or may not improve EIG/competence;novelty is unproven.
