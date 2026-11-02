# Completed policy-training runs

|Version / variant|UbuntuPC completed runs|LabPC completed runs|
|---|---|---|
|V1|contextual_gradient,seed102: **20,575updates**, **41,150,000examples**, 3:59:57, **0.6997s/update**|original_shared7_top3,seed101: **29,198updates**, **58,396,000examples**, 3:59:58, **0.4931s/update**<br>contextual_gradient,seed101: **26,335updates**, **52,670,000examples**, 3:59:57, **0.5467s/update**|
|V4|**NONE VERIFIED**|formal_utility_routing_shared7_top3,seed101: **29,464updates**, **58,928,000examples**, 3:59:30, **0.4877s/update**<br>formal_utility_routing_shared7_top3,seed102: **29,587updates**, **59,174,000examples**, 3:59:30, **0.4857s/update**|
|V5-A auxiliary ON|V5-A auxiliary ON,seed101: **7,714updates**, **15,428,000examples**, 3:59:01, **1.8591s/update**|**NONE VERIFIED**|

Chargedtime includes declared setup/readiness/serialization, excludes recordedvalidation; mean is amortizedcost per final-policy update, not optimizer-kernel time. Historical V1/V4 origins are INFERRED_FROM_LOGGED_GPU + contemporaneous original launch/output identity, not fileholding machine; literal hostname fields absent. V5 host directly observed and recorded. V1 original and contextual are DIFFERENT variants, not duplicate seed101.

Partial OFFcontrol and stopped seed1 are in training_runs.csv;diagnostic32updatearms excluded. Four-expert priorV4 is explicitly excluded from this1shared/7routed/top3 lineage.
