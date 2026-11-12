# Native sBOED correctness

Audit date: 2026-11-12. Exact object: frozen V2Actor candidate, native iteration16001, actor SHA256 `90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108`. This is not V2.1, RR-MoE, or the later routing pilot.

Source aliases on LabPC: `B=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_icml_20261109`; `A=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_internal_ablation`; `V=/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101/source/src/research/vendor/vsoed`. Locations below use these exact aliases and line numbers. Measured CPU evidence: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/moe_hec_v2_code_audit/verification_02/MEASURED_AUDIT.json`. Five deterministic unit tests plus actual-checkpoint tests; 114 protected files hash-preserved, zero optimizer steps, zero new observation simulations, zero GPU jobs. No native objective or historical model/source was changed.

## Objective and update implementation

The native training objective is full-horizon VARIATIONAL information gain, not exact EIG or training-time PCE. With gamma1, incremental r_t=log q_{t+1}(theta|observed prefix_{t+1})−log q_t(theta|prefix_t), prior at t0; the sum telescopes to terminal log q30−log prior. Actor loss is−mean Q_live(x,actor(x)) over10,000 replay trajectories×30 stages=300,000 state-action rows. Critic MSE target is r_t+Q_target(next history,actor_target(next history)); transition0 gives full TD factor1; terminal critic target sums final reward slots. No next action is used after the30th observation.

Per native iteration:1,000 new T30 episodes,10,000 replay trajectories from1M capacity,5 critic steps,5 steps per each of30 posterior networks,1 actor step; target Polyak tau.1,LR decay.9999,exploration decay.9999. Candidate completed2,000 additional actor updates,10,000 critic updates,300,000 posterior optimizer steps,2M new episodes/60M observations. V2 phase B200 freezes history only; C1800 trains history too. Live native posterior/critic weights update independently; frozen actor providers and anchors do not. All network/provenance distinctions are explicit in the expert report.

Actor parameters remain differentiable through continuous commanded model queries, expert MLPs, router and fusion. Belief extraction and frozen SG action are intentionally under no_grad. Frozen opportunity-critic weights do not train, but its input derivative can flow to commanded actions. Prospective searchsorted node indices have no continuous selection gradient. These are explicit implementation choices, not unnoticed expert/router detachments.

## Chronology and leakage tests

Native form_actor_input and prefix_inputs match exactly at t0/3/12/24/29. Replacing every unseen design and observation by arbitrary large values leaves the packed prefix AND action bitwise unchanged. Actor.forward accepts x only: true simulator theta is not an input. Expert preparation samples approximate-belief latent variables; native posterior/reward fitting legitimately uses simulator theta in training. This is separate from policy-time information. These tests cover the implemented input path, not every possible scientific dataset leakage scenario.

## Physical model and numerical checks

Natural log signal m(theta,a)=log(.1+sum_{j=1}²1/(||theta_j−a||²+1e−4)); y~Normal(m,.5²). theta has4 coordinates, design2,source prior iid standard normal; design bounds[−4,4]². Fisher uses noise multiplier4, consistent with variance.25. Physical output a=a0+4delta−a0 abs(delta),delta∈[−1,1],a0∈[−4,4], preserves bounds. Source deterministic function, expert query function, analytic Jacobian and scorer were compared on saved values.

| Check | Maximum absolute discrepancy |
|---|---:|
|Native signal vs expert signal|1.66533453694e-16|
|Native vs literal Gaussian log likelihood|7.33015870225e-10|
|Analytic vs autograd parameter Jacobian|2.22044604925e-16|
|Sum rewards vs terminal log q−log prior|9.53674316406e-07|
|Frozen scorer vs literal positive+11-negative complete trajectory|0|

The native likelihood's tiny normalization discrepancy reflects float32 sigma/normalization constants in a float64 comparison; it cannot explain a .0086-nat observed gap and is not identified as a physical-model bug. The audit originally demanded an unnecessarily strict1e−12 there, then used a precision-appropriate1e−8 assertion without changing native code. Reward telescoping is float32; evaluator likelihood accumulation is float64.

## Evaluation and RNG

Saved main bank:theta[2000,4],noise[2000,30],prior contrasts[1M,4]; its SHA matches EIG_RESULTS.json. Frozen internal full scores are bitwise identical to the original candidate score vector. Positive is included once,denominator1,000,001; literal2-trajectory11-negative test evaluates660 negative observation-density terms and matches the frozen scorer exactly. No contrast subsampling is used in original results. The literal audit uses existing observations, not newly simulated trajectories.

Original evaluation creates banks from dedicated np.random.default_rng seeds distinct from preparation/training; common theta/noise/bank pair policies, and changed policies resimulate observations. The diagnostic bank is separate from the2,000-case bank. HEC-v2 is deterministic; routing_uniforms are for historical stochastic-policy comparators, not categorical selection in V2. No new physical random draw or rollout was generated by this audit. Saved evaluations remain development results, not an untouched final test after repeated architecture development. MCSEs omit seed variability,contrast bias and systematic model error.

Saved live critic and each first posterior Adam step count: **80005** / **80005**, consistent with5 steps per accumulated16,001 native iterations. This audit executed zero optimizer steps. No native sBOED correctness defect was detected in the tested objective, prefix, likelihood, chronology or estimator paths.

Locations: `V/pg_vsoed.py:222–274` (packing),`:608–721` (posterior,reward,critic,actor,target updates); `V/post_approx.py:497–540` (NLL fitting); `V/models.py:43–120` (physical model/likelihood); `B/source/native_reward.py:2`, `B/source/normalized_reward.py:2`, `B/source/frozen_evaluator.py:6`, `B/source/evaluate_v2.py:108`.
