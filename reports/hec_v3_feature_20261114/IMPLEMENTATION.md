# Replacement V3: V2-preserving feature experts and routing

This replaces the retired standalone-head/categorical-selection V3. It is based on the frozen successful V2 checkpoint, not the failed V3 checkpoint. Four experts retain the original sBOED roles. The mixed policy initially reproduces V2 exactly. Single-expert modes initially reproduce V2's corresponding one-hot pathways; they do **not** initially reproduce full V2 performance.

## Shared foundation

Observed history is float32[...,117]: stage one-hot30, previous29 designs(58 coordinates), and previous29 observations. Unobserved entries are explicitly masked. Frozen original V2 history encoder produces128 features; frozen belief, SG provisional policy and commander provide model-derived context and20 commands split4/6/6/4. No actual latent truth or future observation is an actor input.

The shared history contribution and the pretrained V2 decoder are retained and frozen. No new independent action head is trained for each expert. The old per-expert first/response projections are preserved; the original matrix-multiply order is retained for exact numerical initialization.

## Expert feature split

| Expert | Frozen V2 query retained | New private input | Private network | Output |
|---|---|---|---|---|
| E1 region | Posterior-component attention and region/anchor packet | Per-component normalized location4, log scale4, mass1, region command4:13 coordinates | 13→64 SiLU→64 SiLU→128; posterior-attention pooling |128-d decision innovation |
| E2 hypothesis discrimination | Predictive probes, tilted hypotheses, analytic discrimination12 and learned V2 adapter | Discrimination12, command6, provisional design2, posterior mean4, log scale4:28 |28→64 SiLU→64 SiLU→128 |128-d decision innovation |
| E3 parameter sensitivity | Posterior-weighted Fisher/Jacobian geometry and sensitivity/anchor packet | Squashed Fisher upper triangle10, direction4, command6:20 |20→64 SiLU→64 SiLU→128 |128-d decision innovation |
| E4 remaining horizon | Predictive/opportunity probes, sequential packet and prospective response | Prospective response12, probe designs6, frozen critic contrasts3, remaining budget1, command4:26 |26→64 SiLU→64 SiLU→128 |128-d decision innovation |

Every private network's final output layer starts at zero; its hidden layers are separately initialized. All inherited V2 query modules remain frozen in this pilot. Each new expert has different information and computations. The zero initial innovations preserve the parent policy; they are not evidence of learned specialization. Initial output-layer gradients are nonzero. Hidden-layer gradients become nonzero once output weights move from zero; this standard residual initialization is explicitly tested.

The legacy packet is first64 + response64. Each expert's projected contribution and new innovation live in a common128-dimensional preactivation space. The decoder computes

`pre = W_history*s + b + sum_k g_k*(W_first[k]*first_k + W_response[k]*response_k + innovation_k)`.

The implementation keeps V2's equivalent concatenated GEMM for the legacy part to preserve floating-point identity; the old slot projections and shared decoder are frozen. It does not claim all concatenation has disappeared. The new innovations use a weighted sum in the common space. The final V2 ReLU,128→2 layer, tanh and bounded residual action map remain unchanged.

## Router

The original V2 coordinator/response gate remains a frozen prior on these same query/decoder semantics. A new147→64 SiLU→4 correction network starts with zero final weights/bias. Its147 inputs are history128, remaining budget1, contribution magnitudes4, standalone proposal offsets8, and pairwise proposal distances6. Proposal-derived routing inputs are detached so experts cannot backpropagate through that route-input path.

`g = softmax(10*tanh((V2_raw_logits + correction)/10))`.

Zero correction gives exactly V2's original gates. Uniform means weights1/4 across all four feature contributions. This is a soft collaborative feature MoE; it does not sample a physical action from one expert. All four query pathways run in mixed mode. There is no sparse-compute claim.

Each expert is also a complete independently callable policy with common providers/decoder and its own query. Standalone exports contain exactly one current expert; other current experts and the router are not executed. This satisfies standalone execution through a shared decoder, rather than providing four separately trained action heads.

## Learning protocol

Use the unmodified native full-horizon variational EIG objective, original critic/posterior updates, physical bounds and chronology. No new fitted routing-value MLP, imitation objective, KL penalty, balancing loss, online optimization, or forced expert-use targets are added. The original native actor loss is `-mean Q(history,actor(history))`.

Bounded pilot:50 native expert-innovation updates with original routing,50 router-correction updates with experts frozen,100 joint updates. An original-V2 continuation control gets200 native updates. Each arm therefore gets200,000 new episodes/6,000,000 observations,200 actor steps,1,000 critic steps and30,000 posterior steps. New actor optimizer groups use fresh Adam at the current terminal LR per phase; the control inherits its optimizer. The candidate target initially copies its online actor so frozen projection and command-offset buffers agree; otherwise lagged V2 weights converted to frozen buffers would remain permanently mismatched. The control retains its native target lag. Replay, critic and posterior originate from the same V2 runtime. Equal native updates do not imply equal parameters, FLOPs, optimizer histories or target initialization.

Acceptance requires measured wins over its own uniform and all four fixed pathways, original V2, matched control, DAD, vsOED, RL-sBOED and ScoreBED on the mandatory original2,000-case/1M-contrast evaluation, with paired uncertainty and multiplicity correction. Frozen baseline vectors may be reused only after byte/value/checkpoint/protocol verification of the prior exact reproduction certificate; V2 is additionally rerolled and rescored. Standalone ablations use the same trained expert weights and are not independently optimized standalone baselines. A short one-lineage pilot cannot establish ICML-level superiority.

Until measured, useful conditional specialization and gains over uniform are hypotheses.


## Measured validation

The complete original V2 score vector was reproduced exactly over2,000 episodes and1M contrasts: mean 12.92935261 nats, maximum score/action errors0. This is initialization preservation,not an improvement.

Registered parameters: 169,382; trainable innovation/router parameters: 65,476. Fixed provider weights stored as buffers are additional. Standalone execution, prefix masking, finite gradients and hidden-layer connectivity passed. Candidate and control each passed one native update with1 actor/5 critic/150 posterior steps.

The failed V3 active source directories now contain retirement notices; immutable source and checkpoints remain available for historical comparisons. No old campaign was restarted.
