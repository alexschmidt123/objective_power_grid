# DeepSeekMoE-sBOED-RC method

Starting point: validated teacher-free DeepSeekMoE-v1 architecture and deterministic top3 training path, not the newer stochastic IR score-function mechanism. Original IR gradient estimates add discrete score-function variance, while logged routed influence and uniform-routing differences did not establish utility. This motivates a controlled contribution hypothesis, not a proven explanation of its EIG shortfall.

One new scalar parameter a:
g(a)=0.25+3.75 sigmoid(a)
u=z+shared(z)+g(a) sum_{e in top3(softmax(router(z)))} p_e expert_e(z).

Selected softmax scores retain their published mass. No selected-gate renormalization is introduced. One shared expert, seven routed width8 SwiGLU experts, top3 activation, shared 16-dimensional history encoder and design head remain. Total8435 parameters, activated6899: one more than v1. Experts produce features, not independent policies.

Initialization uses a fresh seed101 network,32 training-only full-horizon trajectories (calibration seed7126100), and g0=clip(sqrt(RMS(shared)/RMS(routed)),1,3). This conservative square-root adjustment does not equalize norms; the information objective may reduce the gain. Observed initial g0=1.529314 approximately; readiness/calibration.json is authoritative. No official data or historical trained weights initialize it.

L=-mean(full-horizon DAD sPCE)+0.001 valid detached-frequency balancing. All weights and the scalar jointly use existing Adam5e-5, betas(.8,.998),0.98 schedule and global norm5 clipping. Exact v1 deterministic top3 membership, piecewise pathwise gradients; no discrete score-function estimator is added. Away from rank ties the selected membership is locally constant; gradients through selected gates, experts, histories and gain remain. No claim that membership is differentiable.

Seed101, B2000,L2000,H30,14400 charged training seconds, no episode cap,50k reference updates; latest eligible checkpoint selected. Validation is excluded from main training and reported separately. Smoke/calibration/conditional diagnostics are separately accounted. One candidate, no hyperparameter sweep.

Deployment is a frozen forward pass only. Simulator, posterior references and evaluation are offline. No teacher, competitor checkpoint, online candidate search or optimization.

Sources: [DAD](https://proceedings.mlr.press/v139/foster21a.html), [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/). RC is our adaptation, not a published contribution of either paper.
