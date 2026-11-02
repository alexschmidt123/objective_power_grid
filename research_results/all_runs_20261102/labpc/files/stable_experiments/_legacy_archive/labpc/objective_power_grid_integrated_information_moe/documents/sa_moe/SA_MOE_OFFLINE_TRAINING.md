# Offline objective and predeclared experiment

One seed101 configuration; no hyperparameter sweep. All optimization offline. Total remaining-sequence information objective uses sPCE: U=logp(y_{t:T}|theta0,h)-logmeanexp_{l=0:L}logp(y_{t:T}|theta_l,h). It is a finite-contrast lower bound, not an unbiased true EIG. Main trajectories include allT30 stages. Objective shares established DAD/sPCE machinery; no novelty is claimed for sPCE itself.

Warm-up trains full-horizon sPCE. After100updates, learned history embeddings are initially clustered by farthest-seededkmeans;100router CE updates bootstrap context differentiation without permanent expert labels. No action-diversity or equal-use penalty. At updates500,2000,5000, four fresh training history prefixes at stages4,10,18,26 are used. Continuous rejuvenated conditionalSMC approximates the posterior,refined8192particles. Each expert first action is forced once; the current policy then continues through stage30.32paired conditional rollouts/expert and8192contrasts estimateQ and pairedMCuncertainty. No pretrained teacher or critic. This is an approximate conditional information bound, with SMC and finitecontrast errors explicitly retained.

r_e=softmax((Q_e-maxQ)/max(.1,2maxSE)). If more than5%of branch estimates are within.1of the contrast ceiling, targets stay uniform/unresolved. This is noise handling, not a posterior mixture. Router CE fitsr. Each expert receives an additional pathwise remaining-horizon information gradient weighted by competence across contexts; log-domain per-expert normalization prevents rare-expert weights from disappearing. All experts have forced-first-action gradient exposure. Self-confirming collapse remains a risk and is measured, not assumed solved.

Single control uses the same contexts, posterior states, full-horizon objective, simulator randomness and nominal mainupdates/minibatches. Four conditional branch evaluations match the MoE conditional simulation work. Actual runtimes and extra router/SMC work are reported separately. Parametercounts differ0.19%. No claim of equal compute without measured ledgers.

Main/control each receive600seconds of phase allowance and at most12000updates, Adam.0004/clip5, batch8,L16384. New worker total1800second safety ceiling; indivisible complete native batches can overshoot and actual time is reported. This is ADDITIONAL COMPUTE, not an IC matched-budget improvement. No inherited model or training cost. Readiness native probe.62924seconds plus measuredtesttime and unmeasuredstartup/smokeoverhead are separately disclosed. Sourceauthoring excluded from experimentalruntime.

Independent devseed1013:32trajectories/mode, L16384, learned, random-uniform routing and matchedsingle. Development resolution can saturate atlog16385; do not interpret it as exact EIG. Final checkpoint is latest phase-eligible, never primaryEIGselected. Officialevaluation seed1001 is used only afterfreeze:2000trajectories,L1000000,true-likelihood evaluator unchanged. Consumed1001 is comparison data, not untouchedconfirmation.


## Completed seed101 development attempt

Main8801updates/595.049541seconds; control11316updates/595.036776seconds, selected by latest eligible phase checkpoint. Worker measured1192.352808seconds for both training phases and development. Separate readiness/test/latency costs are retained; not all subprocess startup/smoke overhead was measured, and no GPU-active-time claim is made.

Both use the same nominal phase ceiling, but simulation work differs: main34,852,876,080parameter-stage evaluations versus control44,547,803,136, plus shared conditional-posterior construction recorded separately. The control reuses main-generated conditional contexts; their physical generation happens once. Attribution of shared context-generation walltime is not separately measured (it is a subset of the main5.235672second total competence-cycle walltime). Do not interpret nominal phase equality as exact end-to-end compute equality. No pretrained model costs are inherited.

Development information estimates are below prior policies; this configuration is NOT established to improve EIG. Pre-clipping gradient spikes and only3competence-refinement cycles versus8801mainupdates are documented limitations, not proven causal explanations. No posthoc retuning or best-EIG checkpoint replacement was performed. Official2000/L1M/1001evaluation remains in progress.

Held-out competence: zero of4contexts has a2SE-resolved best expert. This is an unadjusted exploratory MC rule, excludesSMCbias, and cannot establish robust specialization. Full targetvalues and paired errors are saved. Future claims would require more held-out contexts, adjusted uncertainty and independent training seeds.

Deployment timings: CPU2threads, histories at stages4,10,18,26,200timed repetitions each after20warmups, totalhistory-packing+encoder+router+selectedexpert0.176-0.255ms for main. These are timing samples from training-prefix contexts, not held-out information results. No numerical posterior update exists in the deployed history-encoder model. Simulator calls were actively forbidden in this benchmark.
