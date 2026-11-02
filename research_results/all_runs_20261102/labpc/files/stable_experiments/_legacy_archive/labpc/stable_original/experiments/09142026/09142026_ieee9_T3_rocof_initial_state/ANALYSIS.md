# Carried initial-state effects on IEEE9 signed endpoint RoCoF

## Scope and timing

This diagnostic uses the production IEEE9 swing equations with one scalar signed RoCoF observation at 3.5 s after each stage begins. It computes [frequency(3.5)-frequency(3.475)]/0.025 in Hz/s. Pulse amplitude is 0.05 pu at physical bus 1; duration remains in [0.2,3] s. No measurement noise is added to response curves; the comparison noise scale is 0.005 Hz/s. Initial latent parameters are either the prior midpoint or 128 uniform prior samples with seed 101, held fixed within each sequence.

There is no extra gap between recording windows: stage starts are 0,3.5,7 s, with observations at 3.5,7,10.5 s. A shorter pulse ends before its recording window ends; this unforced evolution is part of the stage. Starting a new pulse immediately after the previous pulse ends would be a different, potentially overlapping measurement protocol. This audit does not implement that interpretation.

The root launchers' default max-RoCoF observation remains unchanged. This is a diagnostic of a proposed observation variant, not a completed migration of training and inference.

## Correct state-dependent observation model

Let x=(delta,omega) denote all three rotor angles and all three angular-speed deviations; theta=(M,K) denotes static latent parameters. For a fixed window W=3.5,

    x_(k+1) = Phi_theta(W; x_k,d_k)
    y_k = h_theta(x_k,d_k) + epsilon_k

where h is the endpoint frequency difference above. Only x_1 is equilibrium. Later initial states depend on all preceding designs and theta. Measurement noise enters the likelihood and future design choices, not the physical state directly.

The implemented dynamics are delta_dot=omega and

    M_i omega_dot_i = P_mi - sum_j B_ij sin(delta_i-delta_j)
                     - (D_i+K_i/(2*pi)) omega_i + a_i p(t;d).

At 3.5 s the pulse has ended for every allowed duration. The observation is therefore the residual network/damping response. Angle differences affect electrical coupling; speed deviations affect damping and subsequent angle evolution. A common offset added to every angle has no effect in this model. Initial frequency at the observed bus alone is insufficient to determine the subsequent observation.

## Numerical results

At current duration 2.0 s, the same midpoint parameters produce:

| Previous durations (s) | Stage | Signed RoCoF (Hz/s) | Difference from equilibrium |
|---|---:|---:|---:|
| None | 1 | -0.00731302 | 0 |
| [0.3] | 2 | -0.01071105 | -0.00339803 |
| [0.8] | 2 | -0.00464591 | +0.00266711 |
| [1.5] | 2 | -0.00863521 | -0.00132219 |
| [0.3,0.6] | 3 | -0.00460470 | +0.00270832 |
| [0.3,1.2] | 3 | -0.01084325 | -0.00353024 |
| [0.8,1.2] | 3 | -0.00614315 | +0.00116986 |

For example, changing the previous duration from 0.3 to 0.8 s changes the next observation by 0.00606514 Hz/s at current duration 2.0 s, or 1.21 times the specified noise SD. Across prior samples and valid current durations, the 95th percentile absolute carry-versus-equilibrium difference after [0.3,0.6] is 0.01288695 Hz/s, or 2.58 noise SD. This quantile is a descriptive statistic over the chosen sample/grid, not a confidence bound.

## Separating carried angles and speeds

At the midpoint, after previous durations [0.3,0.6], the stage-3 initial angles are [0.275536,0.276967,0.276567] rad and initial frequency deviations are [0.0047619,0.0058551,0.0057625] Hz. Their common angle level is unobservable through the swing coupling; their differences matter.

For current duration 1.0 s:

| Initial state | RoCoF at 3.5 s (Hz/s) |
|---|---:|
| Full carried state | -0.00311096 |
| Reset angles and speeds to equilibrium | -0.00581926 |
| Carried angles, zero speeds | -0.00403636 |
| Zero angles, carried speeds | -0.00489386 |

The angle contribution relative to equilibrium is approximately +0.00178290 Hz/s and the speed contribution +0.00092539 Hz/s. Their sum is almost the full shift, consistent with locally linear dynamics. The reset and partial-state cases are counterfactual diagnostics, not legitimate replacements for state carry.

Local sensitivities at this same history and duration, obtained by central differences, are:

| Initial-state component | Observation derivative |
|---|---:|
| Machine 1 angle | -1.520588 Hz/s per rad |
| Machine 2 angle | +0.536894 Hz/s per rad |
| Machine 3 angle | +0.983694 Hz/s per rad |
| Machine 1 frequency deviation | -1.360372 Hz/s per Hz |
| Machine 2 frequency deviation | +1.044950 Hz/s per Hz |
| Machine 3 frequency deviation | +0.222995 Hz/s per Hz |

These are local derivatives of the endpoint observation, not universal response coefficients or initial-time accelerations. For scale, an isolated 0.001 rad machine-1 angle perturbation predicts about -0.00152 Hz/s change locally. Equal angle shifts cancel; the numerical common-angle invariance check passed.

## Do histories change peaks and valleys?

At fixed midpoint parameters, they mostly do NOT: they translate the curve vertically. The variation of the carry-minus-equilibrium offset over common feasible durations is between 3.6e-9 and 2.2e-7 Hz/s in the tested histories, far below the 0.005 Hz/s noise scale.

This follows from the near-linear fixed-window approximation:

    x_next approximately A_theta(W) x + b_theta(d)
    h_theta(x,d) approximately c_theta x + g_theta(d).

The incoming-state term is approximately independent of current duration because the endpoint time is fixed. Thus memory can be large without moving the curve's extrema. Small angle differences make linearization plausible for these trajectories; this is an empirical finding for these settings, not a global nonlinear-system theorem.

Latent parameters DO change the curve: from equilibrium on the feasible stage-1 grid, the 5th/50th/95th percentiles of the duration giving the largest signed response are 0.20/0.40/0.813 s over the 128 prior samples. The midpoint maximum is at 0.48 s on the 0.02 s diagnostic grid. These are response maxima, NOT EIG-optimal durations. The earlier 0.1 s table placed the sampled maximum at 0.5 s because its grid was coarser.

The increasing-duration constraint removes early peaks after a long previous pulse. Apparent differences in constrained maxima between histories can arise solely from different feasible domains. They must not be interpreted as physical movement of a peak.

## Inference and design consequences

Each candidate theta must carry its OWN state under the actual applied duration sequence. The posterior predictive mean is h_theta(x_k(theta,history),d), not h_theta(equilibrium,d) and not a response from the evaluator's true state. Existing particle-state propagation has the appropriate structural form, but adopting this new sensor requires explicit integration into all objective, gradient and metadata paths.

If the physical initial state at the beginning of the entire episode is known, later state uncertainty is induced by theta uncertainty and the known action history. If that initial state is unknown, its uncertainty must be modeled jointly with theta or constrained by additional measurements. Assuming equilibrium for an actually disturbed system risks biased parameter inference. A scalar observation does not independently identify six static parameters plus an arbitrary unknown initial state in one step. A broad common-angle component can be removed by a reference convention because it is not identifiable here.

Winding curves do not establish adaptive or non-myopic superiority. Here latent-dependent peaks offer a possible source of adaptation; history-dependent offsets, posterior changes and feasible-domain restrictions also matter. Several smaller wiggles are below the measurement-noise SD. Non-myopic value would require early probes to improve later expected information through both posterior learning and state shaping. That requires conditional and multi-stage EIG evaluation; this physical response audit does not measure it.

## Verification and limits

The GPU endpoint observation agreed with independent DOP853 CPU integration from the same carried initial state within 2.57e-11 Hz/s at one representative valid third-stage design. The common-angle-shift check passed. This verifies that representative propagation/compression, not all possible initial states or policy gradients. Seven histories, 128 prior draws plus midpoint, and a 0.02 s plotting grid were used; policies still have continuous durations. No policy training was performed, and no claim of improved method rankings is made.

Artifacts: history_duration_observations.csv, state_ablation.csv, latent_parameters.json, carried_states.json, summary.json, run_config.json, carried_state_effect.png, completion.json.
