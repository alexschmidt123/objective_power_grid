# V4-Spec: one matched seed101 pilot

Original V4 files/checkpoints are immutable. New code subclasses its actor without adding parameters, buffers or decision-network capacity. Native full-horizon variational actor-critic training is called unchanged.

Forward architecture:

    observed history117 -> original belief/history/commander -> four specialist256 features
    each256 ->128SiLU -> nonaffine LayerNorm128
    [history117,features512,remaining1,belief14] ->644->64SiLU->4 softmax
    weighted SUM128 ->32SiLU->2 ->4tanh
    history117 ->16SiLU->2 ->.2tanh
    final = clip(expert action + bounded weak correction,-4,4)

Only warmup changes expert gradient exposure. All four experts remain active and forward gates are exactly.25. For an expert f(context;theta), use

    z = f(context;stopgrad(theta))
        + m * (f(stopgrad(context);theta)-stopgrad(f(stopgrad(context);theta))).

Forward z is exactly native f. Shared-context and shared-head derivatives are exactly native. Only expert-parameter derivatives receive detached regime mask m. This is the requested regime-conditioned gradient allocation,not a claim that masked expert gradients are unbiased full-batch native gradients. No actor reward,critic objective,posterior objective or native loss scalar changes. No target expert/action label is supplied. Both original and modified actor use detached replay prefixes and the native remaining-horizon critic,so this does not truncate a simulator/trajectory autograd graph.

Regime thresholds are medians from1024 original-parent TRAINING replay episodes at all30 observed prefixes. Only ds_hist/ys_hist are read. Geometry uses total posterior covariance;discrimination uses predictive disagreement between posterior components; sensitivity uses the smallest posterior-scaled Fisher eigenvalue; opportunity uses remaining horizon and positive contrasts of the existing frozen opportunity provider at five deterministic probes. Overlap is allowed. These are exposure proxies,not optimal-expert labels.

Train one shared200-update warmup and serialize the complete actor/target,critic/target,posterior,optimizers/schedulers,replay,RNG and counters. Clone this exact state into learned and uniform arms. Each receives1800 further native updates. Learned starts from zero router output weights;uniform remains exactly.25 and frozen. Logical budgets each:2000 actor,10000 critic,300000 posterior,2M new episodes,T30,discount1,use_PCE=False. Unique physical training:3800 actor and3.8M episodes because warmup is shared. Preparation and warmup time are charged to both logical budgets;physical costs are reported once. Four-hour per-method cap includes evaluation and serialization. Queue waiting is not compute.

The new bank is locked before training:seed202611185101,2000 cases,1M contrasts,T30. Original V4 learned/uniform and vsOED are evaluated without retraining on this same bank. Original canonical/pilot/replication results are preserved separately. Saved canonical scores are used only for the authorized saturation audit. Its prespecified64-case nested1M/4M reference-policy diagnostic does not select hyperparameters or change the pilot scorer.

Run tools/moe_sboed_v4_spec_20261118.py --task prepare only once,then --task controller. The controller audits gradients and architecture before any formal warmup. It has no automatic restart/search/five-seed expansion. Failures preserve partial outputs. Reports distinguish completed stages from pending training.

Telemetry exposure counts are replay-history positions,not new independent episodes. For exact expert exposure totals,sum the thirty per-stage counts in float64;the aggregate float32 telemetry counter can round after16M. Model-query counters cover instrumented likelihood/Jacobian/semantic-provider calls. Extra regime opportunity-provider row counts are not independently instrumented and must not be reported as zero. CUDA-synchronized policy timing is batch16 amortized,not batch1 latency;common diagnostic/scorer time is separate.
