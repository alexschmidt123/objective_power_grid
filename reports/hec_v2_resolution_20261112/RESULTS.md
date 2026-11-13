# V2 research progress: resolve measurement before redesign

The frozen V2 checkpoint is unchanged. No native objective or trained model is modified. The corrected all-consumer pathway knockout is implemented and validated:after masking an expert's router inputs AND first/response fusion slots,perturbing its removed outputs changes neither remaining gates nor action. Historical packet-only ablations remain untouched.

## Nested information-bound resolution study

Predeclared first512 original development cases,same policies and full trajectories,original1M prior contrast bank plus independent fixed-seed extensions to4M/16M. Positive included exactly once. This is a development diagnostic,not a new untouched final test or an improved model. The512-case1M mean differs from the old2000-case mean because the case counts differ;never compare different case sets as a paired contrast.

| Negatives | Learned V2 bound | Uniform V2 bound | Paired learned−uniform |95% paired CI |
|---:|---:|---:|---:|---|
|1,000,000|13.001177|12.970353|+0.030824|[-0.071793,+0.133441]|
|4,000,000|13.764672|13.675993|+0.088679|[-0.034831,+0.212189]|
|16,000,000|14.184713|14.087997|+0.096716|[-0.051312,+0.244744]|

The old1M score vectors reproduce to2.7e−14 maximum error. Learned scores within.01nats of the bound's ceiling drop from37.50% to3.32%;within.1 drop from51.56% to8.79%. The absolute bound changes substantially with contrast count:the1M estimator was not resolved for these strong policies. This does NOT prove that a useful routing advantage was hidden:all paired difference CIs still include zero. The16M estimate remains a finite lower bound;absolute mean rises another.42nats from4M to16M,so it is not exact EIG. Resolution sensitivity is not model performance improvement.

## Concrete failure of the replacement value selector

Read-only analysis of its existing512 training/128 independent validation teacher histories:no new fitting or simulation. Against a training-mean constant predictor,training R²=.2123,validation R²=−.2896. Validation MSE=.21535 vs constant.16699. It predicts selected advantage+.22384nats over uniform but realizes+.01867,95%CI[−.06142,+.09876]. This particular replacement did not learn generalizable gate advantages. It is not evidence that no routing opportunity exists. Labels are noisy single-intervention outcomes,and repeated deployment differs from teacher continuation;the individual causal contributions of these factors remain unproved.

## Next executed step

A separate independent confirmation is predeclared before generating fresh cases:4096 new primary cases,independent16M bank,one primary learned−uniform test;first512 fresh cases for all four one-hot pathways and existing vsOED,Holm5 secondary family. V2 remains unchanged.4M is a resolution-sensitivity diagnostic,not a second success opportunity. No architecture,seed or checkpoint selection based on confirmation outcomes. Power planning uses development variance,not a claimed effect. Multi-seed training and other external baselines are not covered. See development/moe_sboed/hec_v2_confirmation_20261112/PROTOCOL.md and the independent run's RESULTS.json when complete.

This is a research checkpoint,not a claim that the project is solved or ready for publication. The new evidence argues against discarding V2's original router solely because of the low-resolution result,and against promoting the failed replacement. Source/configuration,models and previous results are preserved. Only compact reports and source go into Git.
