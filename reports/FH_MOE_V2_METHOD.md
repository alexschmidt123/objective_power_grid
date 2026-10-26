# FH-MoE-sBOED-v2: gradient-demand responsibilities

Status: experimental implementation; correctness passed, scientific pilot failed. Not a validated improvement.

## Exact foundation

Only DeepSeekMoE-sBOED-v1 commit13681d7091189a4e68056e6a2864dc58f9d5700b supplies architecture/training infrastructure.
Its six policy, rollout, simulator and evaluator foundation files remain byte-identical.
Original frozen checkpoint SHA256:f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7.
Original configuration SHA256:b2c920b9830babf3cec75a6fc6c0c305e5ae1a1104744d3c9f368101f360d8b0.
The earlier reproduction recovered10.98906589858419±.04479410868494 with exact canonical arrays.
This v2 pilot uses fresh seed98102601, with matched native/control initialization. Frozen v1 is evaluation-only, never training initialization or a teacher.

The policy remains8434 total/6898 active parameters: DAD sum history encoder,16-dimensional features,1shared width8 SwiGLU expert,7routed width8 experts,top3,shared head.
Original selected full-softmax gate scores remain unnormalized among the selected3. This is preserved published-style semantics, not a bug.
The shared branch supplies a common trainable information representation; necessity is an empirical ablation question, not a guarantee from its name.

## One coherent change

Replace transient median-coordinate gradient surgery with persistent full-horizon-gradient responsibilities that train expert features and the same router jointly.
No architectural parameters or new deployed network are added.
For trajectory i:
\[
\ell_i=-\operatorname{sPCE}_{i,T},\quad
u_{it}=z_{it}+E_s(z_{it})+\sum_{e\in S_{it}}p_{ite}E_e(z_{it}),\quad
g_{it}=B\,\partial(B^{-1}\sum_i\ell_i)/\partial u_{it}.
\]
The adjoint g includes remaining-horizon history/action effects. It is derived only from terminal information, not prefix information or an immediate-EIG substitute.
Simulator truth contributes to the offline utility estimator, never to deployed policy inputs. A sample's generating theta is not claimed to be a posterior for a fixed history.

At scheduled stages1,10,20,29, every10updates, two disjoint trajectory streams accumulate decayed expert-exposure-weighted unit adjoints.
For stream a, expert e:
\[
M_{ae}\leftarrow .95M_{ae}+\sum_{i\in a}q_{ie}\widehat g_i,\quad
W_{ae}\leftarrow .95W_{ae}+\sum_{i\in a}q_{ie},\quad
V_{ae}\leftarrow .95^2V_{ae}+\sum_{i\in a}q_{ie}^2.
\]
Here q is the original selected-gate exposure; m=M/W and weighted support=W²/V.
References are admitted only if stream agreement≥.9, both mean-unit-gradient norms≥.1, both support counts≥100, and at least two admitted reference directions have cosine<.8.
These are predeclared numerical/stability heuristics, not calibrated confidence probabilities or posterior ESS.
Seven expert-indexed slots do NOT force seven clusters. Collapsed or unreliable references disable auxiliary training.
Reference updates occur AFTER current responsibilities are formed, so current samples never define their own reference targets.

With detached reliable reference directions c:
\[
r_{ie}=\operatorname{softmax}_{e\in A}(\widehat g_i^\top c_e/.25).
\]
The references currently inherit expert populations from observed routing exposure. This can create a bootstrap limitation when existing populations have no stable distinct demands; the failed pilot is consistent with that limitation.

## Conditional scalar objective and exact gradients

With adjoints, r and coefficients fixed for this update:
\[
L_{\rm local}=\frac1B\sum_{i,e}r_{ie}\,
\operatorname{sg}(g_i)^\top E_e(z_i),\qquad
L_{\rm router}=-\frac1B\sum_{i,e}r_{ie}\log p_e(z_i),
\]
\[
L_{\rm conditional}=L_{\rm DAD}+L_{\rm balance}
+\alpha_E L_{\rm local}+\alpha_R L_{\rm router}.
\]
L_local is a first-order terminal-information surrogate, not a measured counterfactual Q value or another expert-specific information reward.
All7 feature experts may be evaluated for this OFFLINE auxiliary loss; the actual policy rollout and deployment remain sparse top3.
The router remains the existing linear-softmax router. Targets seek history-predictable gradient demands, checked independently in the pilot.

Each coefficient is at most1 and caps its auxiliary contribution at5% of the native gradient norm in EVERY encoder/shared/routed/router/head/remaining parameter group.
Thus two auxiliaries together contribute at most10% per group by the triangle inequality, before the unchanged global norm5 clip.
Reference targets/coefficient derivatives are intentionally excluded. This is the exact gradient of the stated conditional surrogate with frozen references; it is NOT an unbiased gradient of original EIG alone, a global scalar potential through evolving reference state, or a guarantee of finite-Adam-step descent.
The original full-horizon gradient is computed exactly ONCE and added exactly ONCE. No stochastic-routing score-function term is used.
History z and router logits remain live for auxiliary derivatives; the complete original pathwise history gradient is retained.
Mathematically zero auxiliary contributions do not wake otherwise inactive Adam blocks.

Auxiliary derivative passes run before the original full-horizon derivative, which runs LAST without graph retention, freeing all trajectory buffers.
An initial GPU bug allocated reference buffers on CUDA under a default-device context although reference statistics were on CPU; allocations are now explicit CPU.
A subsequent repeated-pilot OOM exposed unnecessary graph retention. Reordering derivative evaluation fixes graph lifetime without changing the gradient, objective, data, precision or thresholds.

## Tests and actual outcome

24CPU tests plus1native-device regression pass, including conditional-gradient equivalence, finite differences through30stages, live history derivatives, responsibility stability/rejection, exact v1 source, checkpoint/reference roundtrip and inference without gradient/optimizer calls.
The repeated native B2000/L2000/H30 GPU check completed30updates per arm.
The bounded pilot FAILED all scientific plausibility conditions except numerical correctness. No primary training was launched.

## Related primary sources

[DAD](https://proceedings.mlr.press/v139/foster21a.html) supplies the full-horizon amortized design objective.
[DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/) motivates shared and fine-grained routed features.
The gradient-responsibility procedure is a research proposal tested here; it is not claimed as a mechanism established by either paper or as a new proven specialization principle.
Code:src/research/fh_moe_v2_responsibilities.py; native pilot:src/research/fh_moe_v2_pilot.py; tests:tools/tests/test_fh_moe_v2.py.
