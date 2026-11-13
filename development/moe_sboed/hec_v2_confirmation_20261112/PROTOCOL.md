# Independent fixed-V2 confirmation — locked before fresh cases

## Motivation, not a new trained model

The original1M estimator is strongly ceiling-limited. In a fixed first512-case development resolution study, learned-minus-uniform changed from+.03082nats at1M to+.08868 at4M and+.09672 at16M,with16M95%CI[−.05131,+.24474]. Neither a hidden routing benefit nor superiority was established. One-million scores reproduced the original saved vectors to2.7e−14;16M still has finite-bound bias. These findings justify testing the original unchanged router with better measurement,not selecting another architecture from test results.

The failed replacement selector also shows training R²+.212 but independent128-history validation R²−.290 for its paired advantage labels. It predicts selected advantage+.224nats on validation but realizes+.0187vsuniform,CI includes0. This supports poor generalization of that particular learner;it does not prove zero routing opportunity or diagnose the original native critic. No replacement checkpoint is promoted.

## Independent primary experiment

Freeze successful V2 actor90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108 and runtime d38c09167c4936a6a0660d16079ecc3d5e1108c32bdbc840cc8851c1ef7266b1. Generate4096 fresh prior/noise episodes from seed202611128001 and a separate16M iid prior contrast bank from seed202611128002. These cases and banks are not reused development cases. Both policies share cases,noises,contrast bank. Record nested4M/16M terminal sPCE,positive included once. First and only primary test:learned minus uniform at16M,two-sided paired95% t interval on4096 episode differences. Routing superiority only if its lower endpoint>0. The4M estimate is a resolution-sensitivity diagnostic,not another primary success opportunity.

Sample count uses development paired SD≈1.705nats:4096 gives SE≈.0266 and approximately96% normal-approximation power for a true+.1nat difference at two-sided.05. This is a planning approximation,not a guaranteed effect or result. No interim outcome-based stopping,sample-size extension,checkpoint selection,seed search or estimator-count search. Complete all predeclared policies. Maximum2400 scientific wall seconds after the existing GPU lease;no automatic extension.

## Secondary controls

On the first512 fresh cases(predeclared,not selected by outcome),evaluate all four fixed one-hot expert pathways and the existing frozen vsOED checkpoint at the same16M bank. Compare each with original V2 on those SAME512 cases,Holm5 family. One-hot pathways retain shared history/head/SG reference,not standalone retrained expert policies. Historical vsOED is not compute/initialization matched;report that limitation. DAD,RL-sBOED,ScoreBED and multiple independent training seeds are outside this bounded confirmation. Do not claim all-baseline or multi-seed superiority from this study.

## Execution and preservation

No training or optimizer calls;native objective unchanged. Model weights/source/runtime and vsOED input SHA-verified before/after. Validate the corrected all-consumer knockout on saved states and literal nested likelihood tests. A tiny end-to-end smoke uses2 cases and11/31/127 negatives;no scientific conclusions. LabPC source is committed/synchronized through GitHub to a clean Ubuntu branch before launch. Computation runs only on LabPC,under the idle-GPU lease,never interrupting another job. Every output goes to a new TEMP/scratch_runs directory. Large banks/checkpoints/datasets/logs stay out of Git. The source is frozen during execution. This tests a research claim;it does not guarantee an ICML contribution or acceptance.
