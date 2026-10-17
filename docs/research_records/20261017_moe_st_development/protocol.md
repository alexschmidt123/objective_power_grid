# Straight-through DSelect-1 MoE-sBOED controlled development experiment

Only canonical variant `moe`, training seeds 103 and 105. This is a configurable
training-routing mode, not another named variant. Preserve all journal results.

## One intervention
For native relaxed DSelect-1 probabilities p and native hard Top-1 weights h,
use w = stop_gradient(h) + (p - stop_gradient(p)) in the training action.
The forward is exactly one-hot; native >=0 selector-logit ties are retained.
Compute all four expert proposals during training so unselected proposals
contribute to the routing surrogate gradient. Unselected experts receive zero
direct action gradient. Hard inference still executes one selected expert.
This is a biased routing-gradient surrogate; the sPCE estimator is unchanged.
Native load and selector-entropy regularization continues to use RELAXED weights.

## Fixed settings
Original location physics/prior/noise/T30/bounds; four independent64/64 tanh
experts;193-input sum encoder; component clipping5; Adam lr5e-5,
betas(.8,.998); native .98 decay every1000 updates; batch64; L1024.
Gamma1 through30%, native cosine to .001 at95%, unchanged1M clock.
Exactly1M training episodes per seed; final-budget checkpoint selection only.
No pretraining, balancing change, gate change or optimizer search. Other methods
are not trained or changed. Rootrun.sh is the full-experiment entrypoint.

## Diagnostics and evaluation
Native diagnostic validation512 cases/L1M/seed771001 every125k, and additional
saved checkpoints including750k,775k,800k,812544,825024,832k,850k,875k,883200,
900k,925k,950k,975k,1M. Checkpoints fall at the next completed64-episode batch.
Detailed saturation/logit/gradient/actual-step logging throughout training.
Regular native validation is diagnostic and cannot stop for plateau or select
the best intermediate checkpoint. The1M final checkpoint remains primary.

Evaluate both hard and relaxed policies on identical512-case/L1M/771001 draws
at750016,800000,812544,825024,883200,1M. Use the same original seed103 saved
development checkpoint scores and evaluate its relaxed snapshots as needed.
Original seed105 has native hard validation scores at750336/875392, rather than
exactly matched saved weights; disclose this timing limitation and do not launch
an original-setting training repeat to manufacture intermediate snapshots.

Final primary development comparison:2000 paired trajectories/L1M/seed1001,
hard and relaxed, originals103/105 and straight-through103/105. Original
hard scores may be reused only after paired-draw/score identity verification.
Reserved final seed8831001 is never evaluated. Report trajectory-level paired
Monte Carlo uncertainty separately from training-seed uncertainty; two selected
development training seeds cannot establish general superiority.

Report full EIG curves, pre/post800k changes, original root-cause onset rule
(three saved checkpoints each >=.3 below an earlier running peak with paired
95% upper limit<0), and final changes without selecting favorable snapshots.
The onset rule is descriptive, exploratory, and unadjusted for multiple looks.
Compare seed103 failure recovery and seed105 preservation jointly. A promising
result motivates an UNCHANGED five-original-seed replication before adoption;
no such replication is automatically launched. Keep negative results.

## Execution safety and accounting
Serialize two GPU training jobs and evaluation;8GiB disk floor;3h wall cap per
root phase; original native2h optimization safety cap. An incomplete/failed run
is not labeled1M complete; no automatic hyperparameter adjustment or retry.
Record all generated primary episodes,1024 contrast trajectories per episode,
truth-inclusive particle-stage likelihood work, actual runtimes, checkpoint
hashes, source manifest, test costs, and additional diagnostic evaluations.
New correctness training is16 CPU smoke episodes; GPU64-episode forward/backward
has no optimizer update. Test/diagnostic generation is separate from each1M run.
