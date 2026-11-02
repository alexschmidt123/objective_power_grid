# Current MoE-sBOED-v2 final read-only assessment

**Conclusion: the completed current v2 candidate is worse than its matched native-v1, dense, and trained shared-only controls on both saved development splits. Useful conditional expert specialization is not established.**

## Process and checkpoint status

Training completed naturally at8192matched optimizer updates. Process1209684 has exited; GPU observed idle. Saved state DEVELOPMENT_COMPLETE, stop_reason PREDECLARED_HARMFUL_REGRESSION_STOP. The harmful criterion triggered at the final8192 milestone, so this must not be called an early4096stop. No training interruption, restart, optimizer update, source change, or new MoE version was performed by this assessment.

All four8192checkpoint copies are in checkpoints/. SHA256 values match metadata/final_checkpoint_freeze.json, and every checkpoint update field equals8192. All383frozen source/configuration file hashes match.

## Matched EIG

Seed202training;512paired trajectories per evaluation split;262144common prior contrasts per split;30-stage explicit-likelihood sequential sPCE. Values below are mean±MC SE in nats. These are the completed8192results, superseding our earlier1600snapshot.

| Model | Development97402752 | Replication97402753 | Charged training seconds |
|---|---:|---:|---:|
| v2 | 9.566320 ± 0.081829 | 9.681534 ± 0.082647 | 5606.76 |
| v1 | 10.604670 ± 0.075011 | 10.575397 ± 0.076849 | 5235.49 |
| dense | 10.203234 ± 0.075604 | 10.321612 ± 0.072413 | 3626.84 |
| shared | 10.431250 ± 0.081003 | 10.393354 ± 0.083061 | 3883.13 |

Paired differences are computed from saved per-trajectory score differences, with sampleSD(difference)/sqrt512. Every saved mean, SE, and paired summary was independently recomputed from raw arrays and matched to1e-12.

| Paired comparison | Development difference±SE | Replication difference±SE |
|---|---:|---:|
| v2_minus_v1 | -1.038350 ± 0.100208 | -0.893863 ± 0.099360 |
| v2_minus_dense | -0.636914 ± 0.094925 | -0.640078 ± 0.096576 |
| v2_minus_shared | -0.864930 ± 0.100977 | -0.711820 ± 0.102791 |
| learned_minus_uniform | 0.297828 ± 0.068460 | 0.393054 ± 0.078569 |
| full_minus_mask | 2.022502 ± 0.109469 | 1.924153 ± 0.105486 |

development paired v2−v1 approximate95%MC interval: [-1.234757,-0.841943].

replication paired v2−v1 approximate95%MC interval: [-1.088609,-0.699117].

Uniform routing here is a frozen dense averaging intervention over all7routed experts, preserving the learned top3 total gate mass. It changes both support and averaging, not merely weights within a fixed selected set. Routed masking removes the trained routed output without retraining. Positive frozen-ablation effects establish computational dependence; they do not establish superiority over independently trained shared-only/dense policies. No final8192equal-top3-weight result is claimed; our earlier1600weight-only check is intermediate evidence only.

## Context-dependent expert competence

The saved final specialization study uses two separate previously consumed development case sets, seeds97402755/56,N256/L65536 each. Contexts are observed prefix-mean responses with thresholds fitted on the independent predeclared seed97402754. Seven experts are substituted into the weakest selected slot at stages1,10,20,29, preserving slot weight and excluding duplicates in the first two slots. Terminal full-horizon information utility is evaluated; this is a constrained prefix-law intervention, not exact posterior conditional competence.

- specialization_development: 45conditional effects, 0Holm significant; 48expert×context interactions, 0significant.
- specialization_replication: 44conditional effects, 0Holm significant; 48expert×context interactions, 0significant.

Totals: **0/89conditional effects and0/96context interactions** significant after the source protocol's within-split Holm adjustment. No multiplicity-supported meaningful rank reversal replicated. Apparent different expert weights/gradients, masking losses, or an advantage over uniform averaging do not satisfy this competence criterion.

The missing per-case baseline scores for specialization were independently reconstructed onCPU from the exact saved final checkpoint and the already-used development seeds. All89conditional means/SEs matched the original reports to within2.34e-14. No official or reserved randomness was generated.

## Exact limitations

- One training seed202; two independent evaluation splits are not two independent training replications. The result supports a negative conclusion for this candidate/run, not a universal theorem about MoE architectures.
- EIG is a finite-contrast sPCE lower bound atL262144 (ceiling log262145≈12.4767nats), not infinite-contrast EIG. Do not directly compare against official references obtained with different budgets/draws/contrast counts.
- MC SE and nominal normal-approximation intervals measure evaluation-case uncertainty conditional on the frozen models and shared contrast bank; they omit training-seed and contrast-bank uncertainty. Development data and previous milestones were inspected; these are not untouched official confirmation results.
- Holm control is within each stated split/test family, not across every historical milestone and model comparison. Minimum eligible context-bin count16, coarse context summaries, and weakest-slot-only substitutions limit power and coverage. Failure to reject is not proof that every expert lacks competence.
- The uniform-all7 and routed-mask interventions are frozen-policy/off-policy ablations. The separately trained shared-only control is the relevant trained comparator, and it outperforms v2 here.
- The earlier1600snapshot was evaluated while training was active, using low-priorityCPU only; the campaign then completed naturally. This final report uses its subsequently recovered8192checkpoints and saved final scores. No model was promoted or another version launched.

## Checkpoint SHA256

- v1: `7e28b4a09c91303a6e2b4031ae4137a141f7f0684ead0bcafc71d4dff028df3e`
- v2: `d42bbd719a6f9e4a261f7a96a683c24f91ceab74f2e11ff17003bb162791405b`
- dense: `5191192a239194ddd6680085b3e5ffcbc0995704fef15146638cbbc9fe896c3a`
- shared: `307f05e8577ec52834645fef11f5c3d717de7b2728b984c691ad4cdfac876807`

Original campaign: `/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10282026/10282026_moe_v2_context_readout_seed202`. Final assessment artifacts: `/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10282026/10282026_current_v2_readonly_051919/final_8192`.
