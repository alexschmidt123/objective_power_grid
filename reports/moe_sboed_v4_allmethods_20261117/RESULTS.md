# moe-sboed: V4 all-method canonical-bank benchmark

Status: COMPLETE (50/50 rows). No new training.

|Method|101|102|103|104|105|Mean ± seed SD|
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

|moe-sboed minus|Mean paired difference|95% seed-level CI|Holm p|
|---|---:|---|---:|
|moe-sboed-uniform|+0.05722|[+0.00031, +0.11412]|0.0492263|
|moe-sboed-e1|+6.95533|[+5.56308, +8.34758]|0.000783179|
|moe-sboed-e2|+6.43545|[+3.55058, +9.32032]|0.00690991|
|moe-sboed-e3|+9.79598|[+9.38078, +10.21118]|2.92818e-06|
|moe-sboed-e4|+8.68043|[+7.91407, +9.44678]|4.265e-05|
|DAD|+3.04042|[+1.95292, +4.12792]|0.00445366|
|vsOED|+0.54924|[+0.41544, +0.68304]|0.00135228|
|RL-sBOED|+1.73490|[+1.62235, +1.84745]|1.42553e-05|
|ScoreBED|+1.85554|[+1.67651, +2.03457]|5.20822e-05|

- V4 is the proposed moe-sboed;historical V2/V3 remain archived.
- All scores in this report use the exact original canonical bank;new-bank V4 replication remains separate.
- Shared pretrained initialization:continuation-seed variability,not independent end-to-end training.
- E1-E4 execute only one specialist plus inherited common query provider,action head and weak correction;not independently trained policies.
- The shared head was optimized for mixtures;single-expert EIG measures extracted-path competence,not best achievable standalone training.
- Canonical bank informed historical development;this expanded benchmark is exploratory,not untouched confirmation.
- External native objectives/schedules and inherited pretraining differ;not equal-compute superiority.
- Evaluation MCSE conditions on one common contrast bank;finite-contrast bias is not included.
