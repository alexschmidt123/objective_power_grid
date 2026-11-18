# V4 one-hot expert audit

## Confirmed implementation facts

Extraction exactly equals original V4 decoding with weights [1,0,0,0], [0,1,0,0], [0,0,1,0] or [0,0,0,1]. All five checkpoints passed on 8 observed trajectories at all30 stages:1,200 history prefixes in total. Maximum action error is exactly0. Checkpoint hashes are unchanged. The extracted path executes only its selected specialist plus common query provider, inherited action head and bounded weak correction. No hidden router or other current-expert feature is used. These checks establish numerical/execution correctness,not competent standalone decisions.

Code: tools/moe_sboed_v4_allmethods_20261117.py:ExtractedExpert.forward; src/research/moe_sboed_v4_20261116/policy.py:V4Actor.decode; controlled comparison in tools/moe_sboed_v4_expert_audit_20261118.py:diagnostic.

## Confirmed architectural limitation

V4Expert outputs a normalized128-dimensional feature,not its own action. A single shared128->32->2 head is optimized on a weighted sum of four features. Original V4 training does not separately optimize one-hot expert episodes. Native gradient logs show that expert parameters receive positive finite joint-training gradient norms;they are not simply untouched weights. Code:policy.py:37,60,99-100 and hec_v3_study/engine.py:72-95. No confirmed extraction implementation bug was found.

## Supported empirical weaknesses

On the same saved development histories, individual feature RMS is approximately1.0 while mixture RMS is0.53082;individual-to-mixture RMS ratios are about1.89. Action differences from the full policy are1.25,1.42,1.63 and2.21 for E1-E4. The shared head therefore receives different magnitudes and feature directions when one-hot routing is imposed. General action clipping/tanh saturation is not supported as the primary mechanism:measured head saturation fraction is0 on those histories.

A prespecified quarter-scale diagnostic on a separate32-case development bank improves E1/E4 but not E2/E3. E4 changes from4.28288 to7.33223 nats in this low-resolution diagnostic;E3 changes from3.07337 to2.82985. E3 average consecutive-design movement is0.216 versus0.823 for the full policy. This supports sensitivity to readout distribution and weak extracted sequential decisions,but does not establish that scaling alone explains the low EIG. No scaling was selected as a correction or substituted into official results.

IMPORTANT:The diagnostic uses16,384 contrasts,whose lower-bound ceiling is log(16,385),approximately9.704. Its full-policy estimate9.62663 must not be compared as primary absolute EIG with the canonical1M-contrast table. Primary E1-E4 values remain the original unit-weight one-hot evaluations.

## Unresolved mechanisms and recommendation

The relative contributions of magnitude shift,missing complementary directions and history-distribution drift remain unresolved. Feature differences alone do not establish useful specialization. The code and measurements do establish that mixture training did not produce strong standalone policies under the current readout.

The user subsequently authorized training one-hot experts and restricted it to seed101. Four new runs will use the exact same initial V4 actor weights and V2 parent training state as learned/uniform,with the same native2,000-update/2M-episode budget and canonical evaluation. The router and other three experts remain frozen and bypassed. Each selected expert and its copy of the common head train under native full-horizon EIG. This is new trained-onehot evidence,not a relabelling or overwrite of inherited inference ablations. No seeds102-105 are scheduled and no auxiliary objectives are introduced.
