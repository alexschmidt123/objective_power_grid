# Strategy-MoE-sBOED: one-seed controlled pilot

Separate sum encoder -> StrategyManager (previous soft state, stage, SwitchingGate) -> existing DSelect1 -> four independent adaptive experts. Hard inference executes only the active expert.

27 correctness/regression tests passed, including exact always-reroute rollouts and matching gradients. Original MoE entrypoint smoke and Strategy smoke both passed.

| Policy | EIG | Training |
|---|---:|---|
| DAD | 9.766654 | reused reference |
| sum_moe_original | 9.380161 | reused reference |
| sum_moe_slower | 9.829436 | reused reference |
| always_reroute | 10.235537 | same learned Strategy checkpoint |
| always_continue | 4.218981 | same learned Strategy checkpoint |
| learned_hard | 10.235537 | same learned Strategy checkpoint |
| learned_relaxed | 10.238265 | same learned Strategy checkpoint |

The controlled architecture reference is original sum-MoE with the same legacy gamma schedule. The successful slower-schedule result is an additional reference; its routing schedule differs. No second Strategy training was launched.

Three strategy routing modes share one learned-switching checkpoint; not three independently trained methods. Forced histories can leave usual training distribution.

Paired learned-hard differences and conditional Monte Carlo intervals: {'DAD': {'mean': 0.4688823318304369, 'standard_error': 0.05743096238237478, 'conditional_95_interval': [0.35631764556098233, 0.5814470180998915]}, 'sum_moe_original': {'mean': 0.8553761214206635, 'standard_error': 0.05937805624207991, 'conditional_95_interval': [0.7389951311861869, 0.9717571116551402]}, 'sum_moe_slower': {'mean': 0.40610075997183354, 'standard_error': 0.05787546527439462, 'conditional_95_interval': [0.2926648480340201, 0.519536671909647]}, 'always_reroute': {'mean': 0.0, 'standard_error': 0.0, 'conditional_95_interval': [0.0, 0.0]}, 'always_continue': {'mean': 6.016555315586417, 'standard_error': 0.06992169399041322, 'conditional_95_interval': [5.879508795365207, 6.153601835807627]}, 'learned_relaxed': {'mean': -0.002727890531941112, 'standard_error': 0.006636425080439967, 'conditional_95_interval': [-0.015735283689603446, 0.010279502625721224]}}

| Mode | S0 % | S1 % | S2 % | S3 % | Reroute requests % | Actual switches % |
|---|---:|---:|---:|---:|---:|---:|
| learned_hard | 0.000 | 3.466 | 83.605 | 12.929 | 100.000 | 7.139 |
| always_reroute | 0.000 | 3.466 | 83.605 | 12.929 | 100.000 | 7.139 |
| always_continue | 0.000 | 99.900 | 0.000 | 0.100 | 0.000 | 0.000 |

Learned hard average run duration S0-S3: [None, 1.006006006006006, 24.2455, 3.7495]
Observation sensitivity: {'cases': 56000, 'mean_absolute_probability_change_plus_vs_minus': 9.034713800331974e-05, 'fraction_threshold_flips': 0.0, 'perturbation': 'Latest observed log signal +/- one noise SD; stage, preceding active expert, all other observations and designs fixed. No trajectory altered.'}

Per-mode stage_management.csv reports all29 routed stages. gradient_exposure.json, gradient_norms.json and training_phases.json cover the whole training budget. measurement_patterns.json and matched_expert_proposals.json report quantitative geometry under selected and matched histories. These are posthoc diagnostics; unselected experts were not executed by hard policy inference.

Geometry summary: [{'strategy': 'S0', 'selected_decisions': 0, 'selected_metrics': {'radius': None, 'step_distance': None, 'distance_to_previous_measurement': None, 'nearest_true_source_distance': None}, 'matched_history_metrics': {'matched_proposal_radius': 2.481831500868109, 'matched_proposal_novelty': 0.9167665409867325, 'matched_proposal_nearest_source_distance': 2.2300369988044872}}, {'strategy': 'S1', 'selected_decisions': 2010, 'selected_metrics': {'radius': 0.44084742402135624, 'step_distance': 0.44754829790967665, 'distance_to_previous_measurement': 0.44754829790967665, 'nearest_true_source_distance': 0.9069820025454278}, 'matched_history_metrics': {'matched_proposal_radius': 2.925516753304148, 'matched_proposal_novelty': 1.7001064330887257, 'matched_proposal_nearest_source_distance': 2.6126918918882915}}, {'strategy': 'S2', 'selected_decisions': 48491, 'selected_metrics': {'radius': 1.1691810625844297, 'step_distance': 0.7440346611455977, 'distance_to_previous_measurement': 0.5840922664125274, 'nearest_true_source_distance': 0.9737842030860941}, 'matched_history_metrics': {'matched_proposal_radius': 1.1406333048704478, 'matched_proposal_novelty': 0.6377715792236732, 'matched_proposal_nearest_source_distance': 0.9857215706514955}}, {'strategy': 'S3', 'selected_decisions': 7499, 'selected_metrics': {'radius': 0.7612784243690304, 'step_distance': 0.3872002999829799, 'distance_to_previous_measurement': 0.3769149105380972, 'nearest_true_source_distance': 0.8219357761608078}, 'matched_history_metrics': {'matched_proposal_radius': 2.0100349714388828, 'matched_proposal_novelty': 0.8478423677585504, 'matched_proposal_nearest_source_distance': 1.7947157233476028}}]

Semantic interpretations remain unassigned: radius, spatial novelty and nearest-source distance are proxies; the current diagnostics do not identify hypothesis-disambiguation or verification mechanisms. Frequent selection alone is not specialization.

All training used 1million primary episodes, L1024, seed101; common evaluation used2000 paired cases, L1million, development seed1001. Final checkpoint chosen in advance; MSE not optimized or used for selection.

Training seconds: 1843.6. Native evaluation: 1105.6 seconds. Frozen-mode EIG diagnostics exclude additional posterior inference.

Implementation: src/research/strategy_moe.py, strategy_training.py, optional registration and strategy-record serialization in runner.py, tools/tests/test_strategy_moe.py. Original sparse_moe.py, history implementation, simulator, feasible map, likelihood and common evaluator retain their original hashes. Config: learned.yaml; switching init0.99, no duration/switch schedule, separate component norm5 switching-gate clipping.
