# Fixed protocol

Revised authorization supersedes the earlier novelty-only no-go. PRETRAIN_MANIFEST locks source,configuration,inputs and evidence before ANY preparation/main training. One candidate,two sequential arms(stage then candidate),2000 updates each from verified HEC-v2 iteration16001 to18001. Shared preparation200 steps. No extension,sweep or extra seeds.

Both arms:1000 new episodes/update,10000 replay trajectories,1M replay capacity,5 critic updates,5 posterior updates per stage,gamma1,original optimizer schedules where compatible. Each arm adds2M trajectories/60M observations,2000 actor/10000 critic/300000 posterior steps. Preparation uses no new episodes. The two changed native policy expectations and consistent in-policy exploration are documented separately from the unchanged reward.

One new independent evaluation bank:2000 trajectories,T30,1M prior negatives plus each positive once,fixed exact log-sum-exp scorer. Policies:candidate,stage,frozen HEC-v2/v1,SG,Option,Collaborative,ordinary,vsOED,DAD,RL-sBOED,ScoreBED,and initial stochastic policy. Candidate-stage and candidate-HECv2 primary Holm2;secondary V1/SG/Option/vsOED Holm4. Display ordinary paired95%CIs as unadjusted. One lineage;frozen baselines not compute matched. No final-test access.

Separate512 diagnostic cases:256 calibration/256 heldout. Four routing interventions(uniform,stage mean,shuffled gates,shuffled proposal features) Holm4;four fixed-expert comparisons Holm4;four removals Holm4. Every policy alteration resimulates observations and updates history. Calibration never uses validation outcomes. Distribution-shift caveats apply.

Conditional panel:128 heldout joint-generated prefixes,stages3/12/24,two future-noise replicas per retained theta/history pair. One-step expert intervention then common mixture suffix;compare fixed parent-native terminal variational scores,NOT individual conditional MI. Stage-specific upper-third condition thresholds use separate diagnostic calibration histories.18 high/low group advantages Holm18 and9 high-minus-low Welch interactions Holm9;minimum16 independent histories/group. No prior restart or saved future replay.

Seeds fixed in CONFIG_LOCKED;native simulator RNG resumes parent seed101 lineage. GPU lease and duplicate-safe launcher preserve other jobs. Training grace32400seconds including preparation;evaluation at most3600seconds within grace;TERM33000,hard33600. Correctness failures,STOP_REQUESTED or cap retain incomplete/negative results;no automatic restart or extra training. Ubuntu only receives source synchronization.
