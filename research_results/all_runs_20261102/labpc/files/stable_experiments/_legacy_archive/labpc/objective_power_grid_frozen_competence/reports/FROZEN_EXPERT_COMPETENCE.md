# Frozen DeepSeekMoE-v1 expert information competence

Prespecified success: **False**. Replicated meaningful context-dependent expert rank reversals: **0**. Held-out candidate reversals:0;confirmation:0. No optimizer updates,gradient grouping,policy training,teacher or online design optimization.

The original checkpoint SHA256f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7and six v1foundation files were preserved. Original one-shared/seven-routed/top-three/shared-head architecture,T30simulator,prior,likelihood and[-4,4]constraints were retained.

## Design and estimand

Contexts are two stage-specific bins of the mean observed prefix log-signal,split at a median from1024independent fitting histories. Fitting never uses truth,expert IDs,future observations or information scores. At each stage1/10/20/29,the original top-two experts and all three gate weights remain fixed. The third/weakest selected slot is replaced with a target not in the kept two; kept targets are no-ops. Cardinality remains3and routed gate mass is exact. All later observations and adaptive actions are regenerated,not replayed unchanged.

Expert pairs are compared only on the SAME prefixes where both are eligible. Their conditional domain is observable-bin AND common availability. Marginal means from different eligibility populations must not be compared as a global expert ranking. These are joint-law bin terminal utilities,not exact posterior-conditioned fixed-history EIG. Generating theta is used only by the simulator/utility,not as a posterior or policy feature.

Per-phase Bonferroni family316and normal-approximation criticalz=3.777787. Success requires opposite advantages beyond.02nats with simultaneous intervals in BOTH independent phases,interaction interval excluding0,and>=128common-eligible cases in each bin. All planned comparisons retain the conservative family size even when availability leaves some tests unsupported.

Held-out native terminal sPCE:10.80611388+/-0.03702315;confirmation:10.86810669+/-0.03556701. These are fresh development banks atL262144,not the historical official1MEIG result10.98906590. No performance-based tuning or historical seed1001use occurred.

## Learned router on the tested third-slot domain

|Phase|Stage|Context|Mean alternative-minus-native utility|MC SE|Simultaneous interval|
|---|---|---|---:|---:|---|
|heldout|1|0|+0.001515|0.006766|[-0.024047,0.027077]|
|heldout|1|1|+0.004660|0.009250|[-0.030284,0.039603]|
|heldout|10|0|+0.014905|0.011991|[-0.030396,0.060205]|
|heldout|10|1|-0.016867|0.007992|[-0.047058,0.013325]|
|heldout|20|0|+0.003766|0.007591|[-0.024913,0.032445]|
|heldout|20|1|+0.014417|0.008644|[-0.018239,0.047073]|
|heldout|29|0|+0.006481|0.002595|[-0.003322,0.016284]|
|heldout|29|1|-0.001519|0.000886|[-0.004868,0.001830]|
|confirmation|1|0|-0.005529|0.007440|[-0.033637,0.022578]|
|confirmation|1|1|-0.005370|0.006891|[-0.031403,0.020662]|
|confirmation|10|0|-0.008291|0.006943|[-0.034519,0.017937]|
|confirmation|10|1|+0.006446|0.009014|[-0.027606,0.040497]|
|confirmation|20|0|-0.002210|0.006431|[-0.026505,0.022085]|
|confirmation|20|1|+0.002684|0.008701|[-0.030185,0.035553]|
|confirmation|29|0|+0.002717|0.002911|[-0.008281,0.013714]|
|confirmation|29|1|-0.000107|0.001327|[-0.005119,0.004905]|

There are0phase-specific expert-versus-native rows with simultaneous intervals beyond the.02meaningfulness margin. Such isolated effects are not automatically replicated context specialization; pair reversals and independent confirmation decide the declared result.

A negative alternative-minus-native value favors the learned third choice relative to uniformly replacing it by one of the four unselected experts. This does not test top-two competence,all possible expert sets,or a retrained dense/smaller model. Different selection frequencies or actions alone are not interpreted as information competence.

## Interpretation

The current frozen experts do not demonstrate the meaningful,replicable context-dependent specialization required by this intervention design. This is limited to two observed-signal bins and third-slot substitutions; it is not proof that every possible representation/context lacks competence. Reconsider the expert representation or the premise that the current features encode distinct information-acquisition roles. Do not add routing penalties or resume gradient-clustering training. Where intervals remain wide,report inconclusive contrasts rather than selecting noisy maxima.

MCSE/intervals are conditional on each shared contrast bank. Independent confirmation uses different truth/noise/contrast randomness,but two banks do not certify contrast-resolution convergence. Normal-approximation Bonferroni is asymptotic,not an exact small-sample guarantee. Context eligibility can reduce sample sizes and precision. Frozen intervention evidence does not establish causal superiority of a retrained architecture.
