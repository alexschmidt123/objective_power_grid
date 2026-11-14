# Validation

LabPC smoke04 completed in9.108 seconds. Both arms exercised history rollout, all four counterfactual interventions, conditional posterior estimation, frozen native posterior reward, ensemble fitting, router gradient updates, checkpoint saving and final prior-sPCE evaluation. This eight-case smoke evaluation is not a performance result.

Assertions passed: paired noise and exact observation chronology; identical observed prefixes in every branch (existing futures helper); finite native returns; finite gradients for every trainable router tensor; context/expert state unchanged after updates; native critic state unchanged; historical V2/V3 actor/runtime files unchanged; equal simulator observation counts between arms; full evaluation of learned/control/original/uniform/four standalone/V2 policies. Raw and centered expected-value logit gradients agree within1e-6.

Smoke02 revealed a device-context requirement in the frozen vendor posterior: internally created tensors defaulted to CPU. The new experiment now calls that frozen implementation inside torch.device('cuda'), as the prior validated audit did. No vendor or historical source was edited. Smoke03 passed; smoke04 additionally passed exact paired observation and baseline-gradient checks. Failed attempts remain preserved in scratch directories.

The smoke test uses replicated rows solely to exercise ensemble fitting with its fixed64/16/16 split. Scientific runs use96 distinct independently generated trajectories; no smoke-derived weights or scores initialize scientific runs.

Not established by these tests: useful conditional expert advantages, unbiased exact-posterior labels, critic ranking failure, full-horizon performance improvement, multi-seed robustness, or ICML-level novelty. The protocol distinguishes these from code correctness.
