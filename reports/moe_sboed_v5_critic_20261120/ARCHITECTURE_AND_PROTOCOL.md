# New V5 pilot:previous actor,belief-aware native critic

Official methods:MoE-sBOED V5 and MoE-sBOED V5-uniform. One seed101 pair. Previous history-conditioned V5 remains the development reference;belief-refresh V5 remains RETIRED_NO_RESTART.

## Exactly what changes

The DEPLOYED ACTOR is unchanged:ordered-history conditioning,four specialist feature pathways,frozen inherited belief,644→64→4 softmax router,soft weighted SUM128,128→32→2 head,and bounded.2 weak correction. No belief refresh,no expert-slot concatenation,no large policy decoder,no teacher,no candidate-grid action search. All four specialists execute. The actor's initial state exactly matches the previous V5 initialization.

Only the TRAINING CRITIC changes:

`Q_new(history,action) = Q_parent(history,action) + residual(history,action,belief_summary)`

Parent critic:119→256ReLU→256ReLU→256ReLU→1,trainable with inherited weights and Adam moments. Residual:133→128SiLU→64SiLU→1,where133=observed history117+action2+belief summary14. The summary uses the SAME frozen approximate posterior as the actor:mean4,logscale4,correlations6. It contains no true parameters or future observations. This is extra explicit state information for value learning,not a new deployed policy branch.

The residual output layer starts zero,so initial Q values and design gradients match the parent bitwise. Initial residual and all common states match between learned/uniform. Inherited base-critic optimizer states are preserved;one fresh residual parameter group uses the same current LR and decay. Native target Polyak updates include the residual;its frozen belief provider remains fixed. No new TD target,loss,auxiliary reward or supervised teacher. This tests critic capacity and explicit belief conditioning together;it cannot attribute effects separately.

Frozen belief summaries are cached only for EXACTLY equal prefix tensors,checked by values rather than memory addresses. Changed histories invalidate the cache. This avoids recomputing the same fixed inference during five critic steps and actor-gradient evaluation. Cache tensors are excluded from serialization. The cache is memoization,not an approximation or optimizer. Native GPU smoke test:2 actor/10 critic/300 posterior updates,2000 discarded prep episodes,8.69seconds;2 online-critic context evaluations and10 exact cache hits. Preparation is reported and charged to both logical run ceilings. Formal runs reload the original parent/seed,never this smoke checkpoint.

## Scientific hypothesis and limits

The previous diagnostic found noisy critic-ranking disagreement;it did NOT prove a critic bug. Explicit belief conditioning MAY improve native remaining-horizon value estimation and actor gradients. There is no guarantee of reaching13.15–13.30 nats. The primary test is higher absolute canonical EIG over the previous V5 and V4;learned-versus-independent-uniform is secondary. A larger gap due only to uniform deterioration is not success.

## Fair protocol and no online optimization

Original frozen V2 runtime16001 and previous V5 initialization recipe,not a trained previousV5 or retired refresh checkpoint. Same native discount1/use_PCE=False actor–critic objective,200warmup+1800joint,2000actor,10000critic,300000posterior optimizer updates,2M new episodes and60M observations PER arm. Uniform gates exactly.25 and router frozen/bypassed throughout training/target/evaluation;all other training changes matched. Four-hour hard ceiling per arm includes conservatively charged preparation and evaluation. No selective extension,automatic retries or extra seeds. Actual GPU/CPU/memory/query costs disclosed;equal updates do not imply equal FLOPs. Pretrained/shared provider costs are inherited dependencies,not free from-scratch computation.

The critic is NEVER called to choose or refine an action at deployment. The unchanged actor is tested under frozen parameters/no_grad,with optimizer construction,autograd and random-search traps. No test-time posterior fitting,MCMC,gradient action search,rollout search or weight updates. All richer value estimation is offline native training only. Final deployment audits repeat on terminal policies.

PRIMARY evaluation is exactly the original V2/V4/DAD/vsOED canonical bank/scorer:2000 trajectories,1M contrasts,T30,sigma.5,original truth/base-noise/prior/bounds. No replacement bank. Frozen verified control vectors are reused. Terminal2000 checkpoints only;no canonical checkpoint selection/hyperparameter tuning. Paired95% case intervals are separate from training-seed variability;one continuation seed does not establish robustness. Finite-contrast lower bounds have known saturation,which MCSE does not cover. No automatic seeds102–105 or broad architecture search.
