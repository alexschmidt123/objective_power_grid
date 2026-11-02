# MoE expert and switching diagnostics

All results are development-evaluation replication. First learned measurement excluded from expert statistics. Counterfactual proposals/observations are not rollout EIG or proof of specialization.

| Method | Seed | E1–E4 hard usage | Reroute fraction |
|---|---:|---|---:|
| MoE-sBOED | 101 | [0.002310344827586207, 0.2645689655172414, 0.571051724137931, 0.16206896551724137] | None |
| ResMoE-sBOED | 101 | [0.0, 0.0, 0.8846206896551724, 0.1153793103448276] | 1.0 |
| SwitchResMoE-sBOED | 101 | [0.0, 0.0, 0.8554310344827587, 0.14456896551724138] | 1.0 |
| SwitchMoE-sBOED | 101 | [0.13236206896551725, 0.0608103448275862, 0.8068275862068965, 0.0] | 1.0 |
| MoE-sBOED | 102 | [0.19208620689655176, 0.00503448275862069, 0.8028793103448277, 0.0] | 1.0 |
| ResMoE-sBOED | 102 | [1.0, 0.0, 0.0, 0.0] | 1.0 |
| SwitchMoE-sBOED | 102 | [0.27067241379310353, 0.0020689655172413794, 0.727258620689655, 0.0] | 1.0 |
| SwitchResMoE-sBOED | 102 | [0.0845344827586207, 0.0, 0.9154655172413795, 0.0] | 1.0 |
| MoE-sBOED | 103 | [0.4546206896551724, 0.001327586206896552, 0.42274137931034494, 0.12131034482758618] | 1.0 |
| ResMoE-sBOED | 103 | [0.9969827586206897, 0.0030172413793103452, 0.0, 0.0] | 1.0 |
| SwitchMoE-sBOED | 103 | [0.6294655172413793, 0.0, 0.37053448275862066, 0.0] | 1.0 |
| SwitchResMoE-sBOED | 103 | [0.9677586206896552, 0.03224137931034483, 0.0, 0.0] | 1.0 |
| MoE-sBOED | 104 | [0.0008620689655172417, 0.8927931034482757, 0.0008103448275862069, 0.1055344827586207] | 1.0 |
| ResMoE-sBOED | 104 | [0.0, 0.324551724137931, 0.13179310344827586, 0.5436551724137931] | 1.0 |
| SwitchMoE-sBOED | 104 | [0.0, 0.9623793103448276, 0.0, 0.037620689655172394] | 1.0 |
| SwitchResMoE-sBOED | 104 | [0.0, 0.0, 0.0, 1.0] | 1.0 |
| MoE-sBOED | 105 | [0.0, 0.0, 1.0, 0.0] | 1.0 |
| ResMoE-sBOED | 105 | [0.0001206896551724138, 0.9998793103448276, 0.0, 0.0] | 1.0 |
| SwitchMoE-sBOED | 105 | [5.172413793103448e-05, 0.0, 0.9999482758620689, 0.0] | 1.0 |
| SwitchResMoE-sBOED | 105 | [0.0, 0.016344827586206895, 0.0, 0.9836551724137931] | 1.0 |

Per-expert gradients/final20% starvation, router/gate spikes, durations, proposal differences/residual corrections and observation dependence are in diagnostics/*.json. Distinct actions and usage do not establish that learned routing improves EIG. No algorithm was changed in response to results.
