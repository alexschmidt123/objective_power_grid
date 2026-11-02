# MoE-sBOED-v2 development results

Candidate remains unregistered and full LabPC training is not automatically ready. Source and predeclared method are in MOE_SBOED_V2_METHOD and the pilot plan.

Pilot source `4b2bdfceca488f3b42be9d84c015dc8f9cf87a65`; last recorded matched updates 2400. Training seed201, B2000/L2000,T30. Global pilot training seconds reported so far: 5368.379072065698 (aggregate over four policies, not per-model measured wall).

|Update|Policy/control|Development sPCE|MC SE|
|---:|---|---:|---:|
|128|v1|4.279251|0.154929|
|128|v2|4.356184|0.157327|
|128|dense|4.392255|0.156266|
|128|shared|4.262851|0.157388|
|512|v1|8.286645|0.172914|
|512|v2|8.364706|0.179331|
|512|dense|8.271381|0.177645|
|512|shared|8.219434|0.178623|
|2048|v1|9.212239|0.144687|
|2048|v2|9.650032|0.123040|
|2048|dense|9.098036|0.167326|
|2048|shared|9.330386|0.136996|

Early curves use seed97402631,N128/L65536. They are not official EIG or definitive specialization evidence. Fully trained historical v1 is not the stopping reference for fresh models. All reported comparisons retain finite-contrast and single-development-seed limits.

Final held-out/confirmation results and frozen interventions are PENDING.
