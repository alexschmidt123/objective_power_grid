# HEC V3 decision-level MoE

Implementation directory on the existing research Git branch; no new repository or Git branch. Frozen V2 sources/checkpoints remain unchanged.

## Actual action path

Float32 observed history `[...,117]` (stage onehot30,previous29 designs58,previous29 observations29) -> clear all unobserved padding -> three independent COMPLETE V2 policy copies -> proposals `[...,3,2]` in[-4,4]. Router context is observed history117 +remaining budget1 +detached proposal coordinates6 =124. Router Linear124->128,Tanh,Linear128->64,Tanh,Linear64->3 -> categorical weights. Execute ONE proposal,not its average. Add observation noise in the environment and append the new design/observation chronologically.

Router parameters24451; full V3 parameters593917. Each complete expert has189822 parameters,including its OWN V2 fusion/action head and independent parameter storage. Each retains V2's internal feature modules as initialization; these are not shared between the outer decision experts. This is a decision-level MoE over complete V2-derived policies,not a claim that those original inner computations have been replaced or that meaningful specialization has already emerged.

`v3(history,mode='expert',expert=k)` executes only expert k,without the router or other experts. `v3.experts[k](history)` is directly executable by itself. Each saved `expertK/actor.pt` contains an ordinary complete V2 actor state independently usable outside the mixture. Learned and uniform modes use the SAME categorical execution mechanism; only probabilities differ. Default learned deployment is categorical sampling with explicit optional uniforms for reproducibility. Argmax is an explicitly separate mode,not silently substituted during evaluation. All proposals are computed because the router observes them; this implementation does NOT claim sparse-compute savings.

## Native sBOED training

Physical likelihood,prior,chronology,full30-stage posterior objective,critic/posterior updates,replay,target Polyak updates and physical bounds reuse the existing native implementation. One guarded AST adaptation changes ONLY the actor update from sampled-design `-Q(h,a)` to exact categorical expectation `-sum_k p(k|h) Q(h,a_k)`. This gives a valid router gradient without differentiating the categorical draw,averaging actions,or adding auxiliary rewards. The native vendor file is never edited. An unexpected source block refuses adaptation. Proposal coordinates are detached as router INPUTS so experts cannot manipulate selector weights through that path; joint-phase expert gradients still flow through their proposal Q values.

No true theta,future observations,training critic evaluations or optimization enter deployment selection. Expert policies retain the deterministic model-derived queries already present in V2. Training critic evaluations are offline only.

Identical initialization means initial expert Q values are identical and router credit can be zero; that is expected. Independent native expert training is implemented as a baseline path toward differentiation. It does NOT guarantee specialization,ambiguity/precision/planning roles,or a routing advantage. No such labels are attached to these experts before evidence exists. Role-specific curricula are not implemented or implicitly claimed by this version.

## Entry points

- `test_v3.py`:CPU semantic/gradient tests; `--output NEW_SCRATCH --native-step` adds actual V2 checkpoint,full-episode native chronology,one complete native update and trusted runtime roundtrip validation.
- `train.py --run --output NEW_SCRATCH`:locked short pilot in PROTOCOL.md.
- `policy.py`:standalone/learned/uniform/argmax action interfaces and router/joint training phases.

Validation/training outputs and model weights stay outside Git under isolated scratch_runs directories. No Ubuntu compute is launched.
