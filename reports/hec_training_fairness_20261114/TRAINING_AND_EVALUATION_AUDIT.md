# Training and evaluation audit — 14 November 2026

## Verdict

The canonical evaluation is matched. The training is not a controlled, adequately replicated comparison. The 200-update replacement V3 run is a short adaptation pilot, not a fully trained method comparison. It establishes no advantage over V2 or uniform routing. No training was launched by this audit.

## What actually changed in replacement V3

The replacement restores V2 feature collaboration, rather than selecting one independent action as in retired V3. Four original V2 pathways remain. Added private semantic encoders supply residual features: E1 posterior-region tokens (13 inputs), E2 hypothesis-discrimination features (28), E3 Fisher/sensitivity features (20), E4 prospective sequential-response features (26). Each uses hidden widths 64/64 and output 128, with zero-initialized output layers. A new history/proposal-conditioned router correction uses 147→64→4, with a zero output layer. Initial actions and EIG reproduce V2 exactly.

This was conservative adaptation, not full expert redesign: inherited history encoder, commander, expert query cores and decoder remain frozen. Training covered 55,744 private-encoder parameters and 9,732 router parameters (65,476 jointly). Original weighted feature slots and frozen head are retained; concatenation has not disappeared. Routing changes collaborative features before the bounded two-dimensional action. Each expert is independently executable with its own pathway and shared fixed context/decoder; soft routing evaluates all four.

The pilot used 50 expert-only, 50 router-only and 100 joint updates. Thus each new component received only 150 updates. Native actor/critic/posterior updates and generated episodes matched a 200-update V2 continuation control, but trainability, optimizer history, target initialization and compute did not. Candidate Adam was fresh per phase; the control inherited native optimizer state. Candidate actor target was initialized from the online candidate; the control retained the native lagged target. These are explicit confounds.

## Actual training budgets from receipts

| Method | Actual training | New primary episodes | Measured training time |
|---|---|---:|---:|
| V2 final stage | 2,000 actor; 10,000 critic; 300,000 posterior updates | 2,000,000 | 77.7 min |
| Replacement V3 pilot | 200 actor; 1,000 critic; 30,000 posterior updates | 200,000 | Candidate plus control/evaluation: 25.2 min total |
| DAD | 50,000 updates; batch 2,000; 2,000 contrasts | 100,000,000 | 199.3 min |
| vsOED GMM-I | 10,001 actor; 50,005 critic; 1,500,150 posterior updates | 10,021,000 | 113.4 min |
| RL-sBOED | 1,277,888 recorded gradient updates; native SAC/REDQ implementation | 2,000,000 | 457.4 min |
| ScoreBED | 50,000 score + 100,000 policy updates; five 20,000-update restarts; batch 1,024 | 153,610,054 | 113.1 min |

Historical times are measured receipts, not hardware-normalized compute comparisons. Update counts across algorithms are not equivalent. DAD uses 200 billion training contrast trajectories; RL uses 100,000 contrasts per primary trajectory (200 billion total). ScoreBED's total includes initialization and score training, not just selected policy training. Its 10,000 standardization samples are not added again to the recorded total. All five policy restarts must be charged. Schedule completion does not establish convergence.

V2 itself is inherited: collaborative MoE 10,001 updates → specialization-guided 2,000 → HEC-v1 2,000 → V2 2,000 = 16,001 native actor updates. Known generated primary episodes total 16,021,000 (including initial 20,000 replay episodes), or 480,630,000 observations. V3 adds 200,000 episodes. These totals exclude additional frozen-provider/teacher pretraining whose complete cost has not been reconstructed. V2 preparation adds 300 supervised steps at batch 256 using 4,096 training and 512 validation histories. Shared providers and analytic model computations must also be disclosed and costed.

V2 final training froze history for 200 updates and then fine-tuned it for 1,800; other relevant actor modules trained under native EIG. Current V3 instead freezes all inherited modules throughout. Merely extending this pilot to 2,000 updates would match new interaction count, not reproduce V2's training treatment. Full inherited-network fine-tuning requires explicit code work because context uses no_grad and some inherited matrices are buffers.

The baseline vsOED receipt specifies discount 0.9; V2/V3 final training uses 1.0. The native critic discounts next-stage values, while posterior rewards are incremental log-density changes. Therefore this is an actual objective difference. Preserve the native baseline and label any future discount-1 control separately. Earlier collaborative-MoE ancestry also used discount 0.9.

Some generic run-config fields conflict with executed receipts: DAD defaults mention batch 64/1,024 contrasts, while effective settings and training counters confirm 2,000/2,000. vsOED generic defaults similarly disagree with its native executed profile. This audit prioritizes effective receipts, counters and checkpoint provenance over generic metadata. DAD selected terminal update 50,000; vsOED selected 10,001. ScoreBED selected the final EMA policy from restart 4 among five native final restart policies, using separate validation.

## Evaluation: what is fair and what is still missing

All listed policies use the original V2 evaluation: 2,000 identical cases, 1,000,000 contrasts, original rollout uniforms, T=30, noise sigma=0.5, four-dimensional standard-normal source prior and designs in [-4,4]^2. The original float64 scorer uses chunk 16,384 and includes truth once. The saved V2 learned/uniform, DAD, vsOED, RL and ScoreBED vectors were verified exactly; V2 was additionally rerolled/rescored.

Bank SHA256: 52c0c614c4ee3b63d58ccda60714b38b00a2c0ca35564a428fe8cc2ce8d727c5

Uniforms SHA256: 6e56f0a08fe4deeefd7f0977967dba9d2910a4a383f1acfbd1736db3f358033c

Scorer SHA256: 64c67b6547c77f4823b487493ed0703efbd3f04d7308019e875d145505695f2f

| Policy | EIG, nats |
|---|---:|
| V2 learned | 12.92935261 |
| V2 uniform | 12.93794410 |
| New V3 learned | 12.91092652 |
| New V3 uniform | 12.91284680 |
| vsOED | 12.44271971 |
| DAD | 10.31407383 |
| RL-sBOED | 11.19164733 |
| ScoreBED | 10.99883133 |

New V3 minus V2: -0.01843, paired 95% CI [-0.05590, 0.01905]. New V3 minus uniform: -0.00192, CI [-0.05673, 0.05288]. Neither establishes an advantage or a reliable degradation. These intervals describe case variation for one training lineage, not training-seed uncertainty. Current uniform and single-expert runs are inference-time ablations of one checkpoint, not separately optimized uniform/single baselines. Five seeds in an intended configuration are not evidence of five completed independent lineages.

The frozen baseline inventory explicitly says historical_not_compute_matched. V2 has about 190k registered actor parameters plus 15.27M frozen buffer weights/model providers; DAD's deployed policy has 5,170 parameters and vsOED 162,306. Additional belief/model-query/Jacobian/opportunity-critic computation differs substantially. Same evaluation does not establish equal training compute, capacity, information access or inference cost. No online optimization baseline is required by this audit.

## Evidence locations

Machine-readable source paths and SHA256 fingerprints are in AUDIT.json. V2 actual receipts live under scratch_runs/moe_hec_v2_icml_20261109/{CONFIG_LOCKED.json,arms/candidate,preparation}; ancestry under moe_collaborative_soft_queue_20261108, moe_specialization_guided_eig_20261108 and moe_hec_sboed_20261108. DAD's receipt is beside dad.pt in the preserved objective_power_grid_dad_budget_fix_20261021 legacy experiment. RL receipt is scratch_runs/journal_location_T30_2M_seed101_105_20261101T172522Z/runs/seed101/rl_sboed/training.json. ScoreBED receipt is scratch_runs/journal_LF_T30_native_final_20261102T052000Z/runs/seed101/scorebed/models/scorebed_training.json. Native vsOED reference receipt is listed in AUDIT.json. Existing canonical and V3 pilot reports contain per-case statistics and verification certificates.
