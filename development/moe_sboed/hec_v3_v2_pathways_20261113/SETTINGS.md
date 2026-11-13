# New V3: actual settings, based on V2

This replaces the withdrawn three-copy V3. It keeps FOUR distinct V2 query pathways and V2's commander/coordinator computations. No labels based only on random seeds.

| Expert | Preserved V2 computation | Learned modules | Own decision head |
|---|---|---|---|
|E1 geometry|Posterior-component attention and geometry tokens|region|256->128 ReLU->2 Tanh|
|E2 discrimination|Hypothesis-weighted predictive probes and paired discrimination|discriminate,adapter2|256->128 ReLU->2 Tanh|
|E3 sensitivity|Analytic parameter Jacobians and Fisher-like matrix summary|sensitivity|256->128 ReLU->2 Tanh|
|E4 sequential opportunity|Predictive probes,frozen opportunity critic,prospective response features|sequential,adapter4|256->128 ReLU->2 Tanh|

Every head receives history128 +its own first packet64 +its own response64. Its output adjusts the original frozen SG provisional action through V2's map `a=a0+4*tanh(delta)-a0*abs(tanh(delta))`,bounded[-4,4]. No other expert's packet reaches that expert's head. Heads are separate; there is no shared final action head. The initial head copies only V2's history,own first-packet and own response matrices and final decoder. It exactly matches the original V2 one-hot pathway before preparation.

Shared Context:original V2 history encoder,commander,belief model and provisional SG policy,all FROZEN in this pilot. Observed prefix117 float32;unobserved fields zeroed. The original20 commander outputs split4/6/6/4 as in V2. A standalone export includes its own copy of Context plus ONLY its own pathway/head. It requires neither the router nor other experts. Context sharing in the combined model does not imply another expert's output is needed.

Router:V2 coordinator384->64 Tanh->4,plus original response_gate256->4,plus new zero-initialized proposal_gate8->4. Router context is history128+four legacy packets256,plus four responses256 and four design proposals8. Logits use V2's `softmax(10*tanh(raw/10))`. Initial weights match V2 exactly. Learned execution samples ONE expert proposal;uniform execution samples with probabilities1/4 using identical mechanics;single-expert mode runs one expert throughout the episode. No averaged physical designs. All four pathways are computed for learned/uniform selection;no sparse-compute claim.

Verified parameter counts:236296 total,25956 routing. Original V2 has189822 parameters. Pilot native updates train only own pathway/head for each standalone expert;Context stays frozen. Selector phase freezes all four experts and trains coordinator,response_gate,proposal_gate. The native actor loss for categorical routing is `-E_h sum_k p(k|h) Q(h,a_k)`,not the critic value of an averaged action. No critic/optimization is used at deployment.

Training:300 head-only action-distillation steps/expert on4096 parent-replay observed histories,512 heldout,teacher original full V2. This is initialization,not a replacement EIG objective or proof of specialization. Then250 unchanged full-horizon native-EIG updates per expert,250 selector updates,1250 ordinary-V2 control updates. All expert query mechanisms differ by construction;useful conditional advantages still need measurement. No invented diversity,balance or role reward.

Baselines:uniform,all FOUR standalone experts,original V2,simulator-budget-matched V2 continuation,historical DAD,vsOED,RL-sBOED,ScoreBED. Historical baselines are explicitly not compute matched. Step-DAD/online optimization excluded. Acceptance requires statistical gains;implementation alone cannot guarantee them.
