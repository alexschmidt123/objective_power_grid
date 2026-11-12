# One RR pilot, fixed before scientific updates

Location finding,T30,four iid N(0,1) source coordinates,noiseSD0.5,
log(0.1+sum_source 1/(squared_distance+1e-4)),design box[-4,4]^2.
Native variational actor-critic,gamma1,no cost or other policy reward.

Start frozen HEC-v2 candidate iteration16001 under uniform feature coordination.
One shared100-update uniform-Top2 preparation,then learned arm A2000 updates,
then matched uniform arm B2000 updates. Both inherit identical prepared actor,
target,critic,posterior,replay,Adam/scheduler and simulator RNG. The new RR Adam
starts at the inherited scheduled LR with fresh compatible moments; those moments
continue after preparation. Native10000 replay episodes/batch,1000 fresh
trajectories/update,1M replay capacity,5 critic and5 posterior updates/stage.
Full native batches are accumulated in chunks; no stage or episode is dropped.

Preparation adds100k fresh episodes/3M observations,100 actor,500 critic and15000
posterior updates once. Each main arm adds2M episodes/60M observations,2000 actor,
10000 critic and300000 posterior updates. Costs of inherited lineage,shared
preparation,main training,evaluation and disposable checks remain separate.
A12hour training cap and fixed evaluation/hard caps preserve incomplete outputs;
no automatic restart,extra seed,architecture sweep or post-hoc extension.

One new independent main evaluation bank after both models freeze:2000 paired
T30 truth/noise trajectories and1M negative prior contrasts. Positive included
once. Evaluate both RR arms,frozen original and uniform HEC-v2,frozen SG,native
vsOED,conventional MoE,compatible preserved DAD/RL/ScoreBED,uniform deployment and four adaptation
removals. No historic bank is reused as a paired comparator. Primary full-minus
control/originalHEC/uniformHEC tests use Holm3; mechanistic full-minus-uniform
and four removals use Holm5; external contextual comparisons use their own Holm6.
Ordinary95% t intervals are labeled unadjusted; MCSE is conditional on this one
trained lineage and common contrast bank.

A128-case prefix panel at stages3/12/24 uses two fresh future-noise/routing
replicates per case and five next-decision interventions,then normal RR suffixes.
Group thresholds use first64 histories,remaining64 are diagnostic heldout cases;
minimum16 cases/group. Conditional tests use a separate Holm family. No stage
is an expert slot. Full trajectories are resimulated for all changed policies.

All source/config/check receipts and preserved input hashes enter
PRETRAIN_MANIFEST.json. The launcher checks the lock,prevents duplicates and
respects existing GPU leases. Original jobs are never interrupted. Every validated
source/report checkpoint is synchronized LabPC→GitHub→clean Ubuntu ff-only.
Synchronization alone never authorizes Ubuntu computation.

Conditional relative competencies compare each of six adaptation pairs in each
uncertainty group at three stages,with Holm36 correction. Specialization is
conservatively supported only for an opposite-sign low/high relative-contribution
crossover significant in both groups after correction. This diagnostic does not
change the policy objective or gate training.
