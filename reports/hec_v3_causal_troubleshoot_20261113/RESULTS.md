# Frozen V3 troubleshooting: expert conversion versus routing

**Strongest observed result: reconnecting the trained V3 query cores to the original V2 joint fusion/head and inherited routing recovers V2-level historical-bank performance.**

All results use the exact same2000-case historical V2 bank,noise,T30 likelihood and1M contrasts. No training or checkpoint changes. This is diagnostic reuse,not fresh method validation.

| Policy/execution | Estimated EIG,nats |
|---|---:|
|Original V2|12.92935261|
|Current V3|12.80310089|
|V3 fixed E3|12.86149725|
|initial_router|12.83470188|
|argmax|12.81832563|
|critic_greedy|12.83926316|
|block3|12.80350893|
|episode_commit|12.82157659|
|V2_head_bridge|12.93775104|
|prepared_E1|12.72770161|
|prepared_E2|12.83726413|
|prepared_E3|12.84289467|
|prepared_E4|12.79254342|

## Expert design: loss in the decision-interface conversion

Before preparation,the separate heads exactly reproduce V2 one-hot pathways,not the original full four-pathway policy. V2 mixes weighted packet contributions BEFORE ReLU/final decoding;V3 samples a separately decoded design. These maps are not mathematically equivalent.

All four heads imitate the full V2 action with low heldout MSE,but their complete-episode EIG immediately after preparation is below V2.250 native updates per expert yield no statistically supported EIG improvement in this historical-bank test. Pointwise action imitation was an insufficient prerequisite for preserving sequential policy quality. This is an observed design/preparation weakness,not an incorrect-gradient bug or proof that standalone MoE experts cannot work.

The bridge and initial-router ablations use the same inherited routing function and same fine-tuned query mechanisms;they differ in whether information is jointly fused into one V2 design or converted to independent heads followed by categorical execution. Their paired exploratory comparison is:

```json
{
  "bridge_minus_initial_router": {
    "n": 2000,
    "mean": 0.10304915720833097,
    "MCSE": 0.026009150363429233,
    "CI95": [
      0.052041275035662055,
      0.15405703938099988
    ],
    "p_two_sided": 7.692125181566266e-05
  },
  "bridge_minus_original_V2": {
    "n": 2000,
    "mean": 0.008398429696868718,
    "MCSE": 0.01213818383880663,
    "CI95": [
      -0.015406386776898262,
      0.0322032461706357
    ],
    "p_two_sided": 0.4890794338438474
  }
}
```

Bridge minus current V3:+0.134650,paired sPCE95%CI[0.086676,0.182625],Holm10 p4.18e-7. Bridge minus original V2:+0.008398,CI[-0.015406,0.032203];not a demonstrated improvement. Bridge versus initial-router comparison is additional/exploratory. The bridge changes head/fusion/execution jointly and does not isolate a single head-capacity parameter.

## Routing quality

Initial routing,argmax,critic-greedy,three-step commitments and full-episode commitments do not significantly improve current V3 after the existing10-test Holm correction. Thus neither switching frequency nor sampling alone is established as the main cause. Critic-greedy is an offline diagnostic,not a deployment change. Earlier loss/gradient/optimizer checks pass;no coding bug has been established in those tested paths.

## Specific recommendation

The next change should address expert initialization/decision representation:preserve the full V2 decision basis in each standalone expert and specialize through its own query/correction,then validate full-episode EIG before router training. Do not treat low one-step imitation error as preservation of V2. This recommendation is not implemented here;no new training or architecture replacement was launched.

Frozen diagnostic reuse of original historical bank,finite1M prior-sPCE ceiling,one lineage. Critic-greedy mode is offline diagnostic,not production deployment. Bridge is an untrained assembly,not a new accepted model.

Artifacts:`/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_causal_troubleshoot_20261113`
