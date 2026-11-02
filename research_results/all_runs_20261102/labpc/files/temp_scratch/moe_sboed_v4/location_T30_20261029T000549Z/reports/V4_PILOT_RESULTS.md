# V4 pilot results

DEVELOPMENT_UNVALIDATED. Decision: INCONCLUSIVE. One frozen partition and two training seeds; MC intervals condition on training runs and contrast banks.

|Seed|Policy|Common updates|EIG nats|MCSE|
|---|---|---:|---:|---:|
|101|v4_s101|8192|9.458320|0.076470|
|101|v1_context_s101|8192|10.288356|0.064242|
|101|no_role_s101|8192|9.898308|0.065270|
|101|dense_s101|8192|9.414937|0.063348|
|101|uniform_top2_s101|8192|9.422943|0.070003|

Seed101 pairedV4−V1: -0.830035±0.079037;95%CI[-0.9849471819898916, -0.675123617794914].
|102|v4_s102|8192|10.317453|0.067955|
|102|v1_context_s102|8192|10.413125|0.065046|

Seed102 pairedV4−V1: -0.095672±0.074137;95%CI[-0.2409798661857976, 0.049635261836303624].

B32/L131072 adapted-recipe results are separate from supplemental nativeB2000/L2000. No best-EIG checkpoint selection. See final_comparison.json, matched_compute.json, contrast_sensitivity.json and per-seed role/utility diagnostics. No official promotion or causal four-role claim is automatic.
