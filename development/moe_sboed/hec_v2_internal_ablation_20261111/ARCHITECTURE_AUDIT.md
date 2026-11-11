# Verified HEC-MoE v2 architecture and ablation semantics

The successful candidate is V2Actor in the preserved moe_hec_v2_icml_20261109 source,not the V3 stochastic mixture. It contains four128-dimensional feature packets(each64 first-channel plus64 response-channel),NOT four complete experimental-design outputs. Experts correspond to geometry,exact-likelihood discrimination,local Fisher/sensitivity,and prospective information opportunity. Distinct tasks do not by themselves establish useful learned competencies.

Commander:128-dimensional history encoding plus the frozen SG provisional design and legitimate approximate-belief summary form144 context dimensions. A learned commander with stage offsets emits20 commands split4/6/6/4. Commands affect actual model queries and expert computations. This commander remains intact for all six ablations.

Coordination:the384-dimensional history/legacy-feature coordinator plus256-dimensional response_gate produce4 logits;g=softmax(10*tanh(raw/10)). The same g gates BOTH each expert's first64 features and its response64 features. First features enter fusion[0] with the shared128-dimensional history;response_fusion separately maps gated responses into the same128-dimensional hidden decision representation. ReLU and fusion[2] produce a bounded residual delta. Final physical design is a=a0+4delta-a0*abs(delta),with delta=tanh(head) and a0 the unchanged frozen SG reference action.

Single expert k:force g=e_k at every step. Both channels of other experts contribute exactly zero to fusion. Preserve the shared history path,commander,decision head,biases and frozen SG reference. Therefore these are single-expert PATHWAYS in a collaborative model,not separately trained standalone expert-policy checkpoints. Other expert computations may still execute but do not enter the forced fusion. No chronology is assigned to experts.

Uniform:g=(.25,.25,.25,.25) in both channels at every step. This averages neither complete physical designs nor expert selection probabilities;HEC-v2 deterministically fuses weighted features into one design. All interventions generate new observations/history after each changed design. Full policy receives no override and must reproduce historical same-bank scores within1e-8.

Controls change trained feature amplitudes and joint activation patterns,and can be out of distribution. Their EIG is mathematically valid,but a superiority claim is about these frozen interventions,not proof of optimal standalone expert quality or training-level causal specialization. All configurations remain within the same action bounds [-4,4].

Conventional MoE:the existing compatible ordinary.pt is the direct-head policy with four softmax-gated logit heads,followed by its original action transform. It is evaluated without architecture or parameter changes. It has a different training/provider/capacity budget;no compute-matched superiority claim.

Verified target checkpoint receipt:

```json
{
  "actor": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109/arms/candidate/actor.pt",
  "actor_sha256": "90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108",
  "native_iteration": 16001,
  "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109/source/v2_policy.py"
}
```
