# Straight-through replication and reverse ScoreBED crossing

Two separate development investigations, authorized 2026-10-17.
Task A: reuse completed ST seeds103/105 without changes; train101/102/104
using byte-identical scientific source/config to prior ST campaign, exactly1M
primary episodes each, final-budget selection, no early selection or tuning.
Report paired hard/relaxed development EIG2000/L1M/1001, five-seed mean, SD,
minimum, paired per-seed original comparisons, stage utilization and gradient
exposure. Gamma, Adam, regularization, clipping and architecture unchanged.
All diagnostic checkpoints/logging remain enabled exactly as prior runs.

Task B: missing score-bundle102/policy103 crossing only. Frozen bundle102
includes its score config, variables and complete CUDA normalization. Policy
target normalization/config, root RNG and initialization follow native103.
Compare historical103/103 and existing102/102,103/102 crossings. Explicit
ordered labels are score_bundle_seed / policy_training_seed. Retain original
validation-bound correction. Complete model1M including702994 reused score
pretraining episodes,10 policy initialization and296996 policy episodes.
No new score-model fitting, no favorable score seed substitution. Preflight
compares native103 initial parameters/RNG against restored crossing pipeline;
finite gradients and feasible physical actions are required before training.
Score errors on common policy histories include boundary strata; Fisher
identity references use resolution and ESS checks and are diagnostic only.
Crossed bundles can interact with policy normalization; report as a complete
bundle intervention, not isolated learned weight causality.

GPU jobs serialized; 3h per MoE phase,30min ScoreBED phase;8GiB free floor.
Record new versus attributed complete-model episodes separately. Score-error
diagnostic simulations are separate. Preserve all checkpoint hashes and all
original primary results. Reserved evaluation seed8831001 never sampled.
No architecture changes or targeted corrections are authorized by this queue.
