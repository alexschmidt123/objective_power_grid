# HEC-v2 estimator resolution and corrected pathway knockout

User requests solving V2 routing for a rigorous research contribution. This step addresses a newly measured limitation before more training:35.1% of original learned V2's2000 scores are within.01nats of log(1,000,001),49.6% within.1;uniform33.55% and49.85%. This is finite-prior-contrast bound saturation, not software clipping, and does NOT prove a hidden routing advantage.

## Locked study

Freeze successful V2 actor90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108 and all historical source/providers. Two policies:original learned gates and uniform gates of the same checkpoint. First512 original evaluation-bank cases, not selected by outcome. Retain original1M negative bank as prefix; append independently seeded prior draws(seed202611127002), evaluate complete nested1M/4M/16M banks. Include positive once. Evaluate the same full trajectory against every nested bank. Pair the two policies on theta/noise and every negative bank. Reproduce each original1M score to1e−8 before accepting results. Record score means,Monte Carlo intervals,ceiling fractions and paired differences at each resolution.

This is a development estimator-resolution study, not a new untouched final test, training run, or method-selection sweep. A more favorable estimate is not evidence that an architecture was improved. Finite-bound bias can remain at16M; do not call it exact EIG. No automatic increase beyond16M or1800 scientific wall seconds(after lease); no native objective change and no optimizer calls. Small smoke uses2 cases,11/31/127 contrasts,180 seconds and cannot establish scientific conclusions.

## Corrected diagnostic implementation

Add a separate pathway_knockout helper. Zero selected expert's legacy block before the coordinator, response block before response_gate, and both packet channels before fusion; renormalize remaining gates. Preserve original history,commander,SG reference and other pathways. Historical `disable` semantics remain untouched. This is an all-consumer removal of that pathway's outputs, not erasure of all corresponding physical information and not sparse execution. Deterministic tests require arbitrary changes to knocked-out outputs to have exactly zero effect on remaining gates or designs;validate on actual saved V2 states too.

## Preservation and workflow

New source in LabPC clean Git worktree, unit+integration validation, scoped review/commit/push, Ubuntu clean --ff-only pull and identical hashes before launch. Acquire existing GPU lease,wait if busy,never interrupt another job. Outputs only in a new TEMP/scratch_runs directory. Preserve stopped/failed work;no overwrites. All frozen input and source hashes checked before/after. No checkpoints,datasets,large contrast banks or runtime logs in Git. Only source,protocol and compact result reports synchronize. Original actor is never overwritten. The user previously requested immediate execution after validated implementation;this authorization persists for this bounded diagnostic.
