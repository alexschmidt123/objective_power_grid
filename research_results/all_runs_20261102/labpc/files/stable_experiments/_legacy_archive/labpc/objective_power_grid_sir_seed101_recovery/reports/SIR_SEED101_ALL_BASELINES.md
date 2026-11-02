# Recovered SIR ODE seed101 ten-method E1 comparison

**10/10 completed under the documented E1 protocol.** All actual source snapshots, checkpoints, configurations, SIR physics and1Mepisode budgets verified. No new training or GPU evaluation was necessary for this recovered cohort. This is NOT the later stopped strict published-setting restoration, whose four replacement SIR baselines were never trained.

|Method|EIG +/- MC SE|Paired difference vs DAD +/- SE|Paired difference vs vsOED +/- SE|
|---|---:|---:|---:|
|random|4.42945746 +/- 0.06555066|-5.35078673 +/- 0.06517591|-0.98573122 +/- 0.05088187|
|fixed|9.21582283 +/- 0.03485237|-0.56442136 +/- 0.02082006|+3.80063414 +/- 0.04132347|
|dad|9.78024419 +/- 0.03144611|+0.00000000 +/- 0.00000000|+4.36505550 +/- 0.03953831|
|rl_sboed|9.03968339 +/- 0.03379553|-0.74056079 +/- 0.02718613|+3.62449471 +/- 0.03431357|
|vsoed|5.41518868 +/- 0.04086387|-4.36505550 +/- 0.03953831|+0.00000000 +/- 0.00000000|
|scorebed|4.53985236 +/- 0.04036323|-5.24039183 +/- 0.04406892|-0.87533633 +/- 0.02733810|
|moe|9.87840032 +/- 0.02988544|+0.09815614 +/- 0.01362371|+4.46321164 +/- 0.04188353|
|resmoe|9.21562828 +/- 0.03342014|-0.56461591 +/- 0.02338606|+3.80043960 +/- 0.03893481|
|switchmoe|9.86411107 +/- 0.02997246|+0.08386688 +/- 0.01393021|+4.44892239 +/- 0.04232690|
|switchresmoe|9.21501604 +/- 0.03340510|-0.56522815 +/- 0.02337659|+3.79982735 +/- 0.03895088|

## Evaluator and numerical limits

Common2048truth/noise trajectories, evaluation seed1001,1Mcontrast bank, T5. The entire bank files, case arrays and per-method trajectory identities were checked. Scores reproduce the later frozen table exactly. Paired SE is computed from full case-wise differences, not independent-SE addition. MCSE is conditional on each checkpoint and shared contrast bank; it is not training-seed uncertainty. Confirmation seed8831001was not used.

16,384contrasts are inadequate: historical16k-to1M mean shifts are about.093Random,.582Fixed,.792DAD,.501RL and comparably large for the stronger MoE policies. The existing1Mconfiguration was frozen by explicit E1resource scope;4Mwas deferred to E2. The partial128-case1M-to4M probe showed shifts up to.035nats among six methods and does not certify full10-method resolution at.02/.01nats targets. Therefore this is a common **finite-contrast development comparison**, not an unqualified converged true-EIG result. No unrequested4Mevaluation was run.

Corrected ScoreBED uses adapter revision2,pinned native potential_hidden_dim256and verified saved model assets. The old two-channel flat conservative head and its score are excluded. Source, allocations and evaluation identities are in provenance. Native fidelity checks distinguish executable algorithm agreement from original published hyperparameter equivalence.

## Fairness and protocol scope

Each trained method has1Mtotal primary episodes. Residuals have500kshared frozen seed101base+500kcontinuation; base cost is included per-method in COMPUTE.csv and physically shared between the two residual models. Random has no training. Score normalization/initialization, score pretraining and policy episodes are inside its1Mbudget. Optimizer updates differ by native algorithm; actor/critic/posterior/temperature/score counts are reported separately. Equal episodes do not mean equal likelihood work, updates, wall time or hardware compute. Final-budget checkpoints were reused; no selection by the recovered EIG table.

The completed E1protocol includes disclosed resource adaptations: DAD B64/L1024/lr.0003; native REDQ L1024and its recorded replay/update settings; vsOED7810updates with128fresh episodes and its original recorded SIR profile; ScoreBED batch64and oldE1allocation with corrected scalar head. A later strict restoration proposed DAD B2000/L2000/lr5e-5,RL100kcontrasts,vsOED1000fresh/10k replay and revised ScoreBED native batch/allocation. It was stopped before any SIRreplacement. Those are different experimental protocols, not evidence the earlier completed checkpoints are missing. Their supersession notice is retained in provenance. The requested protocol-choice clarification remains separately recorded; this recovered table must not be relabeled as that strict-restoration result.

All ten rows have explicit evaluation status and source/config/checkpoint hashes. No old finite-latent-bank SIR simulator was substituted. The adaptive on-the-fly RK4 benchmark has population500,initial infected2,log-normal rates,Gaussian sigma1infected-count observations,T5,100-unit horizon and.1minimum spacing.
