# V4 moe-sboed: expert audit and Fixed/Random extension

Status: COMPLETE; 60/60 rows, 5/5 expert audits. No retraining or checkpoint changes.

|Method|101|102|103|104|105|Mean ± replicate SD|
|---|---:|---:|---:|---:|---:|---:|
|moe-sboed|12.85611|12.78863|12.97562|12.88366|12.95045|12.89089 ± 0.07491|
|moe-sboed-uniform|12.80587|12.78430|12.87836|12.77321|12.92665|12.83368 ± 0.06616|
|moe-sboed-e1|5.41189|7.26790|6.73875|5.74085|4.51843|5.93556 ± 1.08865|
|moe-sboed-e2|7.15564|8.72175|2.83691|5.86820|7.69472|6.45544 ± 2.26941|
|moe-sboed-e3|3.58498|2.65878|3.16394|3.14958|2.91729|3.09491 ± 0.34267|
|moe-sboed-e4|4.52534|4.19779|3.52178|3.76228|5.04516|4.21047 ± 0.60646|
|DAD|10.31407|10.35344|9.55718|10.49226|8.53541|9.85047 ± 0.82086|
|vsOED|12.44272|12.30648|12.43011|12.26366|12.26530|12.34165 ± 0.08830|
|RL-sBOED|11.19165|11.17205|11.14941|11.12023|11.14663|11.15599 ± 0.02711|
|ScoreBED|10.99883|11.12651|10.91054|11.06600|11.07487|11.03535 ± 0.08325|
|Fixed|8.81587|8.83679|8.88195|8.89972|8.80951|8.84877 ± 0.04020|
|Random|8.21819|8.31255|8.33431|8.26252|8.25823|8.27716 ± 0.04628|

Random has no training; its columns denote policy-RNG replicates,not independently trained checkpoints. Fixed reuses five completed native 50,000-update checkpoints. All rows use the original canonical bank and frozen scorer. Shared pretraining and unequal historical compute remain disclosed in the original report.
