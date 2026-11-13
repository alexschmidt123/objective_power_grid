# Validation before fresh-data confirmation

Source uses the validated nested-score and pathway-knockout operators from hec_v2_resolution_20261112. Their three deterministic tests pass. The512-case completed resolution run reproduces old1M learned/uniform scores with maximum errors1.95e−14/2.66e−14,checks original V2 state unchanged,and preserves all frozen input/source hashes.

The new confirmation driver completed an end-to-end smoke in5.6235s:both primary policies,all four one-hot pathways,and the frozen vsOED checkpoint load and execute on the same fresh fixture cases/bank. Source/model hashes preserved,zero training updates. Corrected knockout validated on actual saved states. Smoke uses2 cases and tiny contrast banks and cannot establish method superiority.

Scientific confirmation is not complete at this checkpoint. The locked protocol commits4096 independent fresh primary cases,512 predeclared secondary cases,16M primary contrasts,one primary routing test and a separate Holm5 secondary family. No model,seed,case or architecture selection is permitted using confirmation outcomes. No training is launched by this driver.
