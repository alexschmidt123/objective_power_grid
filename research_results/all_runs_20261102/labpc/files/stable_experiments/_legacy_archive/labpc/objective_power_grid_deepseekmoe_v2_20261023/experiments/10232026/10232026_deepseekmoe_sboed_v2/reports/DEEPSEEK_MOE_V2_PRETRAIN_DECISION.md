# V2 change selected before optimization

Frozenv1 diagnosticseed1031,256pairedcases/L262144: full10.745050; sharedonly10.714973; routedonly10.616667; equalweights/sameselection10.743507; randomuniformsubset withpreservedgatetotal10.707054. Corresponding differences arewithinMCerror. All7expertsareused,effectivegatecount5.78-7.00. No expert-collapse claim. Routedfunctioncosinesrange-.857to.928, mean.063; theyarenotidentical,butusefulspecialization isunproven. No simulator/gate-normalizationbug wasfound. Original1030development reproducedexactly.

Warmnativefrozenweightpasses:.47434,.47152,.46758seconds. Profilecounts204nonzero,264streamsyncs,30uniqueoperations. This supports testing a dispatch implementation improvement. Late-development gainsarepositivebutnotresolvedat95%MCprecision; no convergence claim.

ONEchange: training-only selected-expertbatchedmatrixmultiplication, toremoveper-expert dynamic indexing/hostsync. Same1shared+7routed/top3,width8,8434parameters,gatefullsoftmaxvalues,alpha.001,DADencoder/norm/head,Adam5e-5(.8,.998),batch2000,L2000,LRschedule,clip5and4hourcap. Inferenceusesunchanged sparseFFNdispatch. No pretrainedweightreuse.

Acceptancebeforetraining: forward/loss/gradients/oneAdamstep equivalent within declaredFP64tolerances, includinginactiveexpertNone-gradient semantics; nativepairedtiming >=5% median speedup. Otherwise STOPwithouttraining. All profiling/test costs recorded separately; no hiddenfreelearning. Fixed latesteligiblecheckpoint, no official-data-basedselection.

Scientificgoal: more reference-schedule updates within sameallowance, NOT a newroutingmethod orclaimofspecialization. Full-budgetshared-only/routed-only/parameter-matcheddensecontrols arequeued aftertheoneprimaryrun; architecturalevidence mustusecomparablebudgets. Uniform/frozenremovalinterventionsremain distinct fromretrainedcontrols.
