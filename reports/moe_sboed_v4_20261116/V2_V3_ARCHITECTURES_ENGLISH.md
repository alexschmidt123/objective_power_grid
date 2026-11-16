# Actual V2 and collaborative V3 MoE-sBOED architectures

This describes the frozen location-finding V2 checkpoint and the successful collaborative V3 used in the completed five-seed comparison. The terminated categorical private-head pilot is a separate experiment;it is not the V3 described here. Dimensions below are verified from the implemented source,not inferred from an intended architecture.

## The main architectural fact

Both V2 and collaborative V3 combine four specialist feature pathways with a strong inherited physical-design policy. They do not make the four current specialists the sole policy. The inherited policy proposes a complete design a0. A shared decision network then uses the specialist information and an unrestricted shared-history representation to generate a bounded correction to a0.

The inherited base is implemented as SGActor. It is itself an older learned policy with its own encoder,expert representations,router,retention mechanism and decision network. Describing this base simply as a plain DAD network would be inaccurate. It is a strong inherited design-policy backbone. Consequently,the current top-level four-expert router is not the only historical routing mechanism executed inside the overall actor.

A zero output from the current correction head gives a_final=a0. Therefore the actor retains a complete history-dependent design policy even if the current specialist correction contributes little. This is a verified structural property. The claim that this property causes the learned-versus-uniform performance tie is still a hypothesis.

## 1. The physical problem and history interface

Location finding has two unknown2D source locations,represented by theta in R4. The prior is N(0,I4). At every stage the actor chooses a2D measurement location in[-4,4]^2. The simulator returns one scalar noisy observation. The horizon is30 experiments.

The exact mean observation model is the logarithm of0.1 plus the sum of the two inverse squared source-to-design distances,with1e-4 stabilization in each squared-distance denominator. Observation noise has standard deviation0.5.

Before experiment t,the actor receives117 float32 values:30 values for the one-hot stage indicator,58 values for29 possible previous2D designs,and29 previous scalar observations. At t=0 only the stage indicator is observed. At later stages only the actual observed prefix is available. The final collaborative V3 wrapper explicitly masks unobserved/future slots before inference. The native training/evaluation history construction also follows experimental chronology.

The deployed actor does not receive the true source positions. True parameters are available to the simulator and to the training/evaluation objective. Parameter values used inside specialist queries are posterior hypotheses or quadrature nodes,not the true hidden parameters.

## 2. Shared inference and context in V2

V2 first performs three history-dependent computations.

First,a frozen inherited SGActor evaluates the history and proposes a0,with shape[B,2]. This is already a complete valid physical design.

Second,a frozen learned posterior provider estimates an eight-component diagonal Gaussian-mixture belief over the four source coordinates. Its outputs are mixture weights[B,8],means[B,8,4] and standard deviations[B,8,4]. A stage-specific posterior network uses the observations and designs available at that stage. Before any observation,the provider represents the standard-normal prior. This is approximate learned Bayesian inference,not an exact posterior calculation.

The provider also constructs64 quadrature nodes[B,64,4] and node weights[B,64]. There are eight axis points for each of the eight mixture components. The specialists use these nodes to query the known likelihood model without observing the true theta.

Third,a history encoder produces a128-dimensional history representation s. The commander receives144 numbers:history128,normalized a0 with2 coordinates,and14 posterior-summary values. The posterior summary contains four means,four log-transformed marginal variances and six standardized cross-covariances.

The commander is144→64 with Tanh,then64→20. A learned stage-dependent20-vector is added before the final Tanh. More exactly,the implementation adds stage_logits[t]-reference_logits[t] to the commander's output. The resulting20 bounded commands are split into4,6,6 and4 values for E1–E4.

Commands specify how an expert interrogates the model or belief. They are not the final physical designs. The commander is shared across the current specialists and can influence their attention,probe directions,query offsets and hypothetical look-ahead.

## 3. E1: posterior geometry

E1 summarizes where the current posterior places probability mass and how that mass is distributed geometrically.

Each Gaussian component produces a9-value token:four normalized means,four log standard deviations and one mixture weight. A command-dependent attention distribution reweights the eight component tokens according to their standardized locations relative to the overall posterior mean.

The learned region network has dimensions9→64→64,with SiLU between the linear layers. Attention-weighted pooling produces a64-dimensional region feature. A frozen copy of the earlier region network produces a64-dimensional anchor feature. V2 keeps the anchor as E1's first feature and represents the learned change as the response:region_feature-anchor_feature.

Thus E1 contributes two64-dimensional channels. The anchor preserves inherited computations;the response permits adaptation. These are feature representations,not independent physical-design proposals.

## 4. E2: hypothesis discrimination

E2 queries whether candidate measurement locations would produce distinguishable observations under different plausible parameter hypotheses.

Its six commands define a direction around a0 and a tilt over posterior nodes. It evaluates the known mean likelihood model at three probe locations:a0 and two opposite offsets. For each posterior node,the legacy discrimination network receives three predicted signals and one attention weight. The network is4→32→64 with SiLU,and attention pooling produces a64-dimensional first feature.

V2 additionally forms two contrasting node distributions by tilting posterior weights in opposite directions. It compares their predicted observations at the three probes. Twelve semantic values summarize separation and predictive uncertainty. Some expressions are KL-like discrimination proxies;they should not be described as an exact full-horizon EIG calculation.

A response adapter receives36 values and maps36→96→52 with SiLU. Its inputs include the12 semantic values,the two four-dimensional tilted posterior means,a0,the six commands,and the current posterior mean and scale. Concatenating the12 semantic values with the52 learned adapter outputs gives a64-dimensional response channel.

## 5. E3: parameter sensitivity

E3 evaluates how sensitive a measurement would be to changes in the unknown source coordinates.

Its six commands define a parameter-space direction and a2D query-location offset from a0. The implementation computes the analytic four-parameter observation Jacobian at the64 posterior nodes. Posterior-weighted outer products form a4×4 Fisher-information-style matrix. The factor4 corresponds to inverse noise variance1/(0.5^2).

The10 upper-triangular Fisher entries undergo a signed log transform. They are concatenated with a normalized four-dimensional direction,giving14 inputs. The sensitivity network is14→96→64 with SiLU. As in E1,a frozen inherited anchor supplies the first64-dimensional feature,and learned_sensitivity-anchor supplies the64-dimensional response.

This is local model sensitivity. It is not proof that the expert alone maximizes sequential EIG.

## 6. E4: sequential opportunity

E4 estimates which measurements may create useful opportunities for the remaining episode.

Four commands define a probe direction,radius and prospective look-ahead orientation. Three probes are placed around a0. A frozen opportunity critic predicts values at the probes. The features include differences relative to the central probe,predictive observation means and variances,remaining horizon and the commands.

The legacy sequential network is14→64→32→64,with SiLU between layers. Its64 outputs form E4's first channel.

V2 also computes a small hypothetical look-ahead. It systematically selects eight belief nodes,uses three synthetic observation quadrature values,and approximates how those outcomes could change subsequent informative probe locations. The resulting12 semantic values contain nine prospective opportunity values and three immediate discrimination values. These are hypothetical model calculations,not future observations from the actual episode.

A20→96→52 SiLU adapter combines these semantic values,critic differences,remaining horizon and commands. Concatenating12 semantic values with52 learned outputs produces E4's64-dimensional response.

The frozen opportunity critic is a model-derived feature provider. The separate native critic being optimized during actor–critic training serves a different role.

## 7. V2 routing

V2 evaluates all four specialists at every decision. It uses a dense soft MoE;there is no top-k pruning or categorical expert selection.

The original coordinator receives384 values:history128 plus the four legacy64-dimensional specialist features. It is384→64 Tanh→4. A separate bias-free linear response gate maps the four concatenated64-dimensional responses,256 values,to four additional logits.

The two logit contributions are added. The implementation bounds them through10×tanh(logits/10),then applies a four-way softmax. The four gates are positive and sum to one.

The routing weights scale specialist feature channels. They are not probabilities used to sample a physical expert policy in V2.

## 8. V2 fusion and final design

V2 retains expert-specific feature slots rather than summing all expert representations into one common vector.

The first fusion path concatenates the unrestricted128-dimensional history embedding with the four weighted64-dimensional first channels. This gives384 inputs to a384→128 linear layer. A second,bias-free256→128 linear layer processes the four weighted response channels. The two128-dimensional results are added,passed through ReLU,and mapped128→2. A Tanh produces a two-dimensional correction delta.

The final design is computed coordinatewise as:

    a_final = a0 + 4*delta - a0*abs(delta)

This formula keeps the action in[-4,4] when a0 is valid and delta lies in[-1,1]. It also makes the complete inherited a0 policy the default when delta=0.

Two strong information paths remain independent of the current routing weights:a0 directly influences the final design,and the shared history embedding enters the correction decoder directly. A learned head can also compensate for constant gate scaling through its expert-specific input blocks. These are architectural possibilities,not confirmed implementation bugs.

## 9. What collaborative V3 adds

Collaborative V3 preserves V2's base policy,posterior provider,commands,four semantic specialist roles,first/response channels and shared physical-action mechanism. It does not replace V2 with independent action selection.

Its first addition is a private128-dimensional innovation network for every expert. Each network has three linear layers and two SiLU activations:

| Expert | Specialized input | Innovation network |
|---|---|---|
| E1 | Component token9 +command4;attention-pooled |13→64→64→128|
| E2 | Discrimination semantics12 +commands6 +a0(2) +posterior mean4 +log scale4 |28→64→64→128|
| E3 | Transformed Fisher/direction14 +commands6 |20→64→64→128|
| E4 | Look-ahead semantics12 +probe coordinates6 +critic differences3 +horizon1 +commands4 |26→64→64→128|

The final linear layer of each new innovation network is initialized to zero. At initial transplantation,this lets the actor preserve the existing V2 action computation. It does not mean these networks stay zero during training.

V3's second addition is a residual router. Its147 inputs are history128,remaining horizon1,four log feature-magnitude descriptors,eight normalized proposal offsets and six pairwise proposal distances. The network is147→64 SiLU→4,with a zero-initialized final layer.

The proposals used in these router descriptors are constructed by sending one current expert channel at a time through the inherited shared decoder and a0 mechanism. They are auxiliary candidate actions for routing descriptors. The deployed collaborative actor does not select or average these physical proposals. They also should not be confused with independently trained standalone policies.

The147-value descriptor vector is detached before the residual-router network. Therefore actor-loss gradients train the residual router but do not flow back into expert/history computations through this descriptor path. Experts and history can still receive gradients through the actual feature-fusion paths and inherited coordinator.

## 10. Collaborative V3 routing and action equations

V3 adds the residual-router logits to the inherited V2 coordinator and response-gate logits. It then applies the same bounded-logit Tanh and softmax.

Its128-dimensional fusion preactivation can be written as:

    pre = W_history*s + b
          + sum_k w_k*(W_first,k*first_k
                       + W_response,k*response_k
                       + innovation_k)

The projection matrices in this expression are blocks of the live shared decoder weights. The router's feature-magnitude descriptors also use live decoder projections. The decoder then applies ReLU,a128→2 action layer and Tanh. The same inherited residual action equation follows:

    a_final = a0 + 4*delta - a0*abs(delta)

Although the innovation vectors have width128,this is not V4's proposed normalized common128-dimensional expert mixture. Collaborative V3 still contains expert-specific decoder blocks,the unrestricted shared-history path and the strong a0 final-action path. Its innovation features do not have V4's per-expert common normalization.

## 11. Learned and uniform training

The completed V3 comparison used200 warmup updates and1800 joint updates. Warmup froze the shared history encoder while training the designated expert/query/fusion/router components. Joint training also unfroze that encoder. Frozen belief/base/opportunity providers and anchor networks remained frozen.

The training objective stayed native full-horizon variational actor–critic EIG with discount1 and use_PCE=False. There were no new balancing,diversity or cost rewards. Per continuation run,the completed schedule used2000 actor,10000 critic and300000 posterior optimizer updates with2M newly simulated episodes.

The independently trained uniform arm fixed the TOP-LEVEL four current expert weights at0.25 throughout training and evaluation. Its new residual router and inherited top-level coordinator/response gate were frozen and received no updates. The experts,fusion and action head remained trainable. The frozen inherited a0 provider still executed its historical internal routing. Thus uniform at the current top level does not mean that every older nested routing mechanism inside the inherited base was replaced with uniform weights.

Joint trainable actor counts were252030 for learned V3 and216374 for uniform V3. The35656 difference is the9732-parameter residual router,24900-parameter inherited coordinator and1024-parameter response gate. Warmup counts were220414 and184758. These counts exclude the separately optimized native critic and posterior networks and exclude frozen-buffer storage.

## 12. What the measured results establish

The five continuation seeds gave learned V3 mean EIG12.91701nats and independently trained uniform V3 mean12.91415nats. Their paired difference was+0.00287nats,with95% seed-level CI[−0.01510,+0.02083]. Adaptive routing did not demonstrate an improvement over trained uniform routing.

The earlier12.95086 versus12.75930 comparison used one learned checkpoint with uniform weights imposed only at inference. It showed sensitivity to changing that checkpoint's gates,not superiority over an independently trained uniform model. Its score vectors are preserved and reproduce the quoted numbers.

The implementation audit verified constant uniform top-level weights,frozen gate tensors,matched initial actor digests and finite nonzero expert/fusion gradients. It did not find a secretly trained uniform router or a disabled uniform expert pathway.

The extracted E1–E4 policies from collaborative V3 use one CURRENT expert plus shared inherited context/decoder. That shared context includes a0,and the a0 provider itself contains historical experts. Therefore these exports are not pure single-specialist networks with every historical expert removed. They were correctly labeled extracted policies with common providers,not independently trained single-expert methods.

The architecture makes routing compensation plausible,but the existing evidence does not establish it as the sole cause. Routing optimization,critic accuracy,expert redundancy and limited useful reweighting headroom remain distinct possible explanations. The V4 pilot specifically removes the strong final-action paths while retaining feature-level collaboration,so it can test the expert-dominance hypothesis more directly.

## Source locations

V2:`experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109/source/v2_policy.py`,with`hec_policy.py`,`query_response.py` and`sg_policy.py` in the same directory.

Collaborative V3:`src/research/hec_v3_study/model.py` and`frozen_feature_policy.py`;training:`engine.py`. The active completed benchmark wrapper/protocol is`tools/moe_sboed_benchmark_worker.py` and`configs/moe_sboed_resume_20261115.json`. Older generic README/25-arm study descriptions are historical and do not describe the finalized10-run queue.
