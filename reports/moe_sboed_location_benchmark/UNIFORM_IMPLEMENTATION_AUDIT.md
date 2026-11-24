# MoE-sBOED-uniform implementation audit

PASS for five learned/uniform pairs, seeds 101–105. Audit reads checkpoint tensors, initialization digests, configurations, source manifests, evaluation arrays and training logs on CPU. No training process, checkpoint, configuration or scientific source was modified. No training or evaluation rollout was launched.

| Check | Verified finding |
|---|---|
| Uniform weights | StudyMoE.compute replaces gates with torch.full_like(gates,.25) whenever arm is uniform, regardless of the mode argument. All 1,200,000 saved evaluation weight entries (5×2000×30×4) are exactly .25. .25 is exactly representable in float32. |
| History/proposal independence | Final top-level gates are constants. Prior logits and proposal descriptors are still computed but discarded. Intervention attributes default to None/False. Official training leaves them untouched; official uniform evaluation installs a constant30×4 quarter-weight table. Internal expert attention, commands, features and final actions remain history dependent; they are not the top-level mixture gates. |
| Router updates | configure freezes router,prior_coordinator,prior_response_gate for uniform. Train-phase pre-step checks assert every frozen parameter has grad=None at every actor update, and phase receipts certify frozen parameter/buffer digests unchanged. All uniform gate-network tensors are bitwise unchanged warmup→final and equal their initialized values: inherited parent gates or exact seeded new-router first layer with zero final layer. |
| Optimizer membership | Adam is constructed over ALL actor parameters, including frozen parameters. Thus router parameters are present in its parameter list but receive no gradient or Adam update. Frozen gradient=None is checked before each actor step. This is not optimizer exclusion. Audit did not deserialize saved optimizer moments or rerun an optimizer step. |
| Expert/fusion training | Four expert groups, decoder.weight,response_weight,action_layer and commander have positive finite gradient norms in all82 saved log entries per run. Joint checkpoints differ from warmup for every expert group, decoder and history encoder. The pre-step hook checks finite gradients every actor step; the log retains only82 snapshots. Frozen anchors, belief/base providers and opportunity provider remain frozen by protocol. |
| History phases | History encoder frozen during200 warmup updates, trainable during1800 joint updates. Commander, stage offsets, expert trainable components and shared decoder are trainable in both phases. |
| Initialization | Exact actor-state digest equality in each of five learned/uniform pairs. Common inherited actor/runtime hashes, new-module seed202611150000+seed, copied target and same fresh Adam recipe. Identical initial parameter values do not mean identical initial actions: uniform deliberately replaces inherited routing. Shared parent pretraining is not independently reproduced per seed. |
| Architecture | Learned/uniform final state keys, tensor shapes and dtypes match exactly. Same four expert classes, private widths13/28/20/26→64→64→128, feature fusion and action head. Encoders enabled in both arms. Saved source hashes match actual frozen scientific source for all ten runs. No unrequested expert-architecture difference identified. |
| Objective | Same engine.train_phase→n.original.train_native path; assertions use_PCE=False and discount=1. No balancing/diversity/cost reward. Critic/posterior update recipe is unchanged. Prior-sPCE is the common evaluation estimator, not the training objective. |
| Simulator/evaluation | Same simulator/likelihood/prior/bounds and packed observed-prefix handling. Same eval bank/scorer hashes, seed202611099206,2000 cases,1M contrasts,T30,sigma.5,chunk16384. Truth and base noise are coupled; policy-dependent realized histories need not match. |
| Budget | Each run completes2000 actor,10000 critic,300000 posterior optimizer updates;2M new episodes and60M simulator observations, same4h maximum. Equal update/simulation budget does not mean equal wall time or trainable count. Learned-only1024-trajectory diagnostic costs an additional30,720 observations;20 extracted expert evaluations add evaluation cost only, not policy training. |

| Phase | MoE-sBOED trainable actor parameters | MoE-sBOED-uniform | Difference |
|---|---:|---:|---:|
| Warmup200 | 220414 | 184758 | 35656 |
| Joint1800 | 252030 | 216374 | 35656 |

Difference: new residual router9732 + inherited coordinator24900 + inherited response gate1024 =35656. Joint adds31616 history-encoder parameters to both models. Counts exclude separately optimized critic/posterior networks.

## Gradient flow

Uniform: native actor loss → action head/shared fusion → four weighted expert feature paths (constant.25) → trainable expert components and shared commands/history. No actor-loss gradient to uniform gate networks. Learned: same paths plus gradient through softmax/tanh and prior/residual gate networks. Proposal/history descriptors into the new router are explicitly detached, so no extra gradient flows back through those descriptors; direct fusion paths still train experts/history. Frozen model-derived providers stay frozen in both.

## Evidence limits and conclusion

Evaluation weights were exhaustively checked in saved arrays. Every historical training-step gate was not saved; training-time constancy is established by the executed source branch, identical source manifests and absence of active overrides, not a nonexistent complete training-gate trace. Frozen state is supported by recorded phase-digest assertions plus initial/warmup/final tensor comparisons. No fresh optimizer step, retraining or GPU job was performed.

No implementation error identified that makes this uniform baseline secretly learned or prevents expert/fusion training. Its strong performance is therefore not explained by those suspected bugs. This audit establishes implementation consistency, not convergence or a scientific explanation of the negligible learned-routing gain.

## Code references

- `src/research/hec_v3_study/model.py:11` — `def configure(self,phase):`
- `src/research/hec_v3_study/model.py:17` — `def compute(self,x,mode='learned',expert=None,collect=False):`
- `src/research/hec_v3_study/model.py:32` — `inp=torch.cat([c['s'],remaining,norms,offsets.flatten(1),distances],-1).detach()`
- `src/research/hec_v3_study/model.py:36` — `if mode=='uniform' or self.arm=='uniform':gates=torch.full_like(gates,.25)`
- `src/research/hec_v3_study/model.py:37` — `if self.gate_table is not None:gates=self.gate_table[c['stage']]`
- `src/research/hec_v3_study/frozen_feature_policy.py:33` — `class FeatureExpert(nn.Module):`
- `src/research/hec_v3_study/frozen_feature_policy.py:82` — `class SharedDecoder(nn.Module):`
- `src/research/hec_v3_study/frozen_feature_policy.py:105` — `def configure(self,phase):`
- `src/research/hec_v3_study/engine.py:47` — `def optimizer(a):`
- `src/research/hec_v3_study/engine.py:78` — `def gradient_check(*unused):`
- `src/research/hec_v3_study/engine.py:87` — `profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.`
- `src/research/hec_v3_study/engine.py:88` — `assert not a.use_PCE and a.discount==1 and profile['critic_updates']==5`
- `src/research/hec_v3_study/engine.py:100` — `def build(seed,arm):`
- `src/research/hec_v3_study/engine.py:151` — `def configure_intervention(model,label,table=None):`
- `src/research/hec_v3_study/engine.py:167` — `for phase,steps in [('warmup',200),('joint',1800)]:phases.append(train_phase(a,out,phase,steps,start,start+12600))`
- `tools/moe_sboed_benchmark_worker.py:14` — `engine.PROTOCOL=copy.deepcopy(engine.PROTOCOL);engine.PROTOCOL.update(methods=cfg['methods'],arms=cfg['methods'],queue=cfg['queue'],inference_ablations=['learned'],benchmark_protocol=cfg)`

Reproduce read-only CPU checks: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/grads/g/g.lin/miniconda3/envs/dellma/bin/python tools/moe_sboed_uniform_readonly_audit.py`. Output goes to stdout. Raw per-seed evidence:UNIFORM_IMPLEMENTATION_AUDIT.json.
