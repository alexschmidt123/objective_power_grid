# V5-A regression attribution
Decision: NEGATIVE completed development result; supported proximal gradient interference and unequal optimization exposure. Ultimate causal attribution remains INCONCLUSIVE without a matched auxiliary-off training control.

## Observed facts
|Evidence|Measurement|Interpretation|
|---|---|---|
|Saved paired EIG|V5-A9.76586593; V4 difference−1.21661020,95%CI[−1.32109474,−1.11212565]|Fixed-policy evaluation regression; not training-seed replication|
|Native-size gradient probes|V5 weighted-aux/main ratio range2.875–80.405, median13.425; all4 dominant;3/4 cosine<0|Reproducible domination and conflict at executed B2000/L2000. Four cases only|
|B16 probes|V5 median ratio0.770;5/16 conflict;5/16 dominant|Gradient scale depends strongly on batch size; do not substitute B16 magnitudes for B2000|
|Adam clones|V5 median on/off update cosine0.9174; norm ratio0.9757; mean route-set change3.28%|Actual optimizer states restored; four discarded B16 one-step clones, not a learning curve|
|Logged late auxiliary norms|mean3817.42, median172.45,95pct4001.66, max324381.20|Heavy-tailed already-weighted auxiliary gradients; clipping5 scales combined vector, not each source separately|
|Late main score/saturation|V5 7.38465/75.49%; V4 7.53209/91.42%|Saturation measured, not proof auxiliary helpful|
|Sampler-only overhead|1422.48s (9.92% charged)|Cannot attribute all throughput difference to posterior overhead|

## Exposure and compute
|Run|Updates|Primary trajectories|Charged seconds|Seconds/update|GPU|
|---|---:|---:|---:|---:|---|
|V4 seed101|29464|58,928,000|14370.14|0.488|HOST/GPU UNVERIFIED|
|V5-A seed101|7714|15,428,000|14341.36|1.859|RTX5080|

Historical V4 host/GPU are UNVERIFIED in staged runtime evidence; see hardware_normalized_accounting.md. Both declare B2000/L2000/T30, max50000, cap14400s, same native Adam/scheduler/clip/balance. V5 achieved26.18% of V4 updates. Hardware provenance is incomplete; microbatch backend, extra replay and auxiliary differ; this is NOT a same-hardware equal-update mechanism ablation. V5 simulator particle stages935,316,789,662 are logged. V4 primary minimum =29464*2000*2001*30=3,537,447,840,000, plus original utility/intervention work; exact V4 total stage counter UNAVAILABLE. Accepted posterior proposal counts, historical CPU-vs-GPU split and isolated microbatch overhead are UNAVAILABLE. Do not substitute acceptance averages for accepted-count totals.

## Ruled out within tested scope
- Architecture/count/routing change: identical8434 parameters/shared1+routed7/top3 and frozen source hashes for architecture. Native main-gradient equality passes both policies; full production source comparison is source_comparison.csv.
- Loss sign or stage normalization implementation error: production maximizes Jfull+0.1*30*C_t via negative backward; truth-inclusive stable logsumexp,Gaussian likelihood and action mapping unchanged. Uniform stage weighting30 is correct for a stage-sum target, but high variance. Signed estimates are not clipped.
- Future/true-parameter leakage into sampler/deployment: prefix-only production test and source inspection pass. Truth remains only positive simulation/likelihood target offline.
- Checkpoint/evaluation mismatch: checkpoint SHA, both benchmark configs, raw score means and paired differences reproduced from SAVED arrays; case/contrast SHA identical to V4. No million-contrast evaluation repeated.
- Checkpoint cherry-picking: last budget-eligible serialized checkpoint selected; 7714final. Stale top-level train_manifest.json still says RUNNING/NOT_STARTED in places; final run_status/checkpoint/evaluation prove completion. Preserve stale metadata as evidence, do not overwrite.

## Ranked supported hypotheses
1. Sparse one-case/one-stage auxiliary has large noisy gradients after30×stage correction, often dominates native gradients and sometimes conflicts. This is directly measured, including native batch size and Adam direction changes. It is the strongest proximal mechanism, not a causal estimate of its EIG cost.
2. Reduced update exposure plus different backend/hardware materially confounds comparison. The native LR remains update-based: fewer updates also imply fewer decays; no scheduler bug was found.
3. Detached finite-SMC surrogate differs from a full differentiable conditional-EIG objective; mathematical gradient bias is known. Posterior sensitivity/missed modes remain possible, but six fixed histories do not establish they caused regression.

## Unresolved
No original gradient-separated training trace; no periodic V5 dev-EIG curve/routing statistics despite requested monitoring; no equal-update auxiliary-off control. Late norm spikes do not establish which posterior case caused a bad update. Saturation stratification in gradient_interference.csv records main norms for saturated/unsaturated cohorts; auxiliary remains one randomly sampled case and cannot be attributed to each stratum as a cohort-average gradient. Future confidence is limited to this trained seed and diagnostic draws.

No production defect was proven. No V5 production code changed, no coefficient tuned, no formal training launched. One authorized32-update/arm local paired diagnostic completed; see tiny_pilot_results.md. It measures immediate continuation effects, not causality for the historical full-training EIG regression.

## Corrected hardware interpretation
# Hardware-normalized provenance and fairness correction
The user states most prior results were generated on UbuntuPC. This report does NOT classify all old results as LabPC runs. Staged copies indicate artifact storage/transfer origin only. Host labels below require an explicit runtime site record or direct observed launch provenance. No LabPC access was performed.

|Run|Training host|GPU|Updates|Primary examples|Charged GPU lease seconds|Updates/s|Examples/s|
|---|---|---|---:|---:|---:|---:|---:|
|V4_formal_seed101|UNVERIFIED|UNVERIFIED|29464|58928000|14370.144592896919|2.050362110800464|4100.724221600928|
|contextual_V1_seed101|UNVERIFIED|UNVERIFIED|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|
|V4_formal_seed102|UNVERIFIED|UNVERIFIED|29587|59174000|14370.46342025767|2.058875843787471|4117.7516875749425|
|contextual_V1_seed102|UNVERIFIED|UNVERIFIED|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|
|native_vsOED_seed101_reference|labpc|NVIDIA GeForce RTX 4090|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|
|V5A_seed101_completed|ubuntu-pc|NVIDIA GeForce RTX 5080|7714|15428000|14341.362072899006|0.5378847532604459|1075.769506520892|

V4 actual host and GPU are HOST UNVERIFIED in the staged bundle: status/config files have counts/timing but no actual-host/GPU record; source code/intended target do not fill that gap. V1 original run configs/runtime records are absent; contextual checkpoint references alone provide hashes, not host/exposure. Native vsOED stagedrun_config has hardware.execution_site=labpc and computer_name=ecen-80r5x04.engr.tamu.edu plusGPU RTX4090: this SPECIFIC reference is recorded as LabPC, irrespective of where most other historical results originated. Its actual training iterations, charged GPU clock, exact checkpoint bytes and original source manifest are not staged; configured10001cycles/1000fresh trajectories do not prove attained values. Its summary reports6805.6383s training_wall, which is not silently substituted for charged leased GPU time. Its top-level/nested selection rules conflict and remain unresolved. V5 observed Ubuntu launch/worker and original compute_clockGPU5080 establish its host/hardware; saved artifacts alone lack a standalone hostname.

Raw nominal settings embedded in benchmark.paper_protocol still mention a1Mepisode pilot although effectiveV5/V4 run budget is50000updates/14400own-host seconds. Use the effective top-levelrun config, actual counts and clock, not stale benchmark documentation.

Cross-run: V5 primary15.428M vs V4 seed10158.928M is only26.18% of update/example exposure, not equal-training/equal-compute. The posterior, likelihood-bank work, backend/microbatch implementation and host uncertainty prevent equal-compute causal claims. No assertion that either GPU is intrinsically faster. Measured run-specific throughput describes these exact recorded workloads only. All-computation FLOPs and GPU-kernel times are UNAVAILABLE. Simulator_particle_stages is a FORWARD-particle-stage counter, not a count of Python simulator invocations; likelihood_evaluations are UNAVAILABLE unless separately instrumented. Primary lower bounds exclude extra route/replay/posterior work. Native vsOED counts cannot be inferred from proposed config alone.

The new ON/OFF pilot holds Ubuntu hardware, initial model+Adam state, primary examples, batch, maincontrasts and32updates fixed. ON performs additional inference/likelihood/objective work, so it is EPISODE-MATCHED and UPDATE-MATCHED, NOT COMPUTE-MATCHED. The32update/600s bound is solely safety. Historical rows and diagnostic rows are separate; pilot arm subrows must not be added again to total. CPU test times are separate; incidental CUDA context duration in first failed test harness is UNAVAILABLE, no benchmark training occurred there.

## Completed bounded paired pilot
# Bounded paired pilot — local optimizer mechanism only
Completed32 updates PER arm from the SAME frozen V5-A final weights/update7714 and identical restored Adam+scheduler state, no original files modified. Same UbuntuPC RTX5080, matched B2000/L2000/T30 and same theta/noise/prior-contrast arrays each update; all32 paired input SHA checks PASS. The pilot starts at a trained V5-A policy, not a fresh V4 initialization. This tests local continuation effects, not the original full-training learning trajectory.

Production primary_backward/auxiliary_backward functions reused. OFF short-circuits the conditional auxiliary in the new diagnostic driver; the original formal CLI still only acceptsalpha.1 and is UNCHANGED. This is not a claim that the unmodified formal CLI exposes an alpha0 switch. No period100 V4 credit event occurs from absolute7715..7746, so no missing event is silently omitted. RNG streams are explicit fresh diagnostic streams, equal across arms, not an exact continuation of the original random sequence.

Total device-lease elapsed131.734s (<600); arm timed costs OFF62.398s, ON68.189s. Initialization/shared validation/serialization difference is1.147s and charged to total. Count64 optimizer updates and128,000 primary-example exposures,64,000 distinct paired examples. Additional truth-only replay, posterior and all likelihood work recorded; same episodes do NOT mean same compute. Forward-particle stages7,732,266,336, with no separate CUDA-kernel timer; times are leased-device wall seconds.

Tiny INDEPENDENT diagnostic validation only:8cases/L4096,seed73491001,never official1001 or reserved8831001. Initial score both8.201013. Final OFF8.194774, ON8.173481; ON−OFF-0.021293±0.168143MCSE across8fixed cases. This is NOT formal held-out EIG, not seed uncertainty, not evidence of reproducible superiority. All0/8/16/32 scores saved; final32 selected by protocol, no best-score selection.

Per-update objectives, main/aux/group gradient norms/cosines, clipping, selected sets, action spread, posterior proposal work and costs are intiny_pilot_updates.csv. No second seed, variant search or extra training. Last common32 checkpoints saved separately astiny_pilot_off_final.pt/tiny_pilot_on_final.pt (diagnostic only).
