# V4 frozen routing ablations: seed101

All six modes use the SAME terminal learned checkpoint. No retraining,reinitialization,feature rescaling,head replacement or physical-coordinate averaging. Shared query,normalization,head and weak correction are unchanged. Original canonical2000-case/1M-contrast/T30 scorer.

|Mode|Weights|EIG nats|MCSE|Reproduction error versus prior vector|
|---|---|---:|---:|---:|
|learned|learned|12.85611|0.03599|0.0|
|uniform|[0.25, 0.25, 0.25, 0.25]|11.43033|0.04963|new uniform intervention|
|e1|[1, 0, 0, 0]|5.41189|0.05889|0.0|
|e2|[0, 1, 0, 0]|7.15564|0.05377|0.0|
|e3|[0, 0, 1, 0]|3.58498|0.04393|0.0|
|e4|[0, 0, 0, 1]|4.52534|0.05406|0.0|

These are inference-only ablations,not separately trained standalone policies. The four trained seed101 one-hot policies and independently trained uniform policy are preserved as different experiments. Case MCSE is not training-seed variability.
