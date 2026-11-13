# Actual-checkpoint validation

Completed on LabPC in12.73seconds. Replayed600 short preparation optimizer steps total; zero native continuation updates.

- Both prepared actor-state digests match historical STARTED.json exactly; E4 candidate digest2205aca5f0dc6bea9bd8479efcccffccb4c80abd34a05a9e578ce26f4478a71c. Complete preparation receipts also match.
- Original and both final episode-level competency errors reproduce historical saved arrays within1e-8. Ordinary-prepared representation errors are identical to original.
- Classifier fixtures verify sign conventions and distinguish preparation gain followed by degradation from no initial gain.
- Full calibration-set loss/gradient observations do not change prepared states, as established by exact digest and receipt matching.
- Historical sources,models,runtimes,reports,logs and score arrays hash unchanged at completion. Reproduced preparation checkpoints are saved in a new isolated scratch directory.

The heldout128-coordinate E4 probe shows no demonstrated preparation gain: original0.1708929553,prepared0.1716078067,final0.1711577675. Adapter-only52-coordinate probes agree. Diagnostic readout calibration improves error,but the control readout-only refresh improves it more without changing policy features/actions. A frozen refreshed adapter experiment is not justified by these results.
