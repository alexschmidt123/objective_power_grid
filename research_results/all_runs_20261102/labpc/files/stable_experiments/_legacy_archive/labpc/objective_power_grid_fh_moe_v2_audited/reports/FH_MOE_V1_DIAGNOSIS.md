# FH-MoE-v1 diagnosis

Both completed runs and exact v1 source13681d7091189a4e68056e6a2864dc58f9d5700b were inspected.
FH-v1's primary LabPC seed101 was11.04404±.04281; Ubuntu seed102 was10.83416±.04301.
Neither establishes reproducible useful specialization.

## Confirmed algorithm and evidence

The old procedure selected the largest-variance coordinate of the current16-dimensional encoder output and split the minibatch at its median, cycling stages1,10,20,29 every10updates.
It did not cluster full-horizon gradient directions. For each expert, two masked terminal-sPCE cohort gradients reconstructed its native gradient, including apportioned original balancing.
When cohort gradients conflicted, the cohort with higher existing selected-gate exposure anchored a projection; the altered expert gradient retained the original norm and positive raw-gradient alignment.
Only routed-expert gradients were changed. Encoder/shared/router/head gradients remained native; the router received no explicit cohort-responsibility target. This provided no persistent expert-context identity.

| Evidence | LabPC seed101 | Ubuntu seed102 |
|---|---:|---:|
| Logged norm5 clipping fraction | 1.000000 | 0.995146 |
| Seconds per update from logged wall | 0.546600 | 0.699591 |
| Encoder RMS range | [0.9963348447188694, 1.014725827933215] | [0.9978049480531083, 1.0102673529751944] |
| Final logged training sPCE | 7.531519 | 7.530356 |
| L2000 sPCE ceiling | 7.601402 | 7.601402 |

Coordinate changes between successive same-stage grouping steps:

- Stage1: LabPC 0.2371, Ubuntu 0.2198.
- Stage10: LabPC 0.2314, Ubuntu 0.3665.
- Stage20: LabPC 0.2116, Ubuntu 0.2943.
- Stage29: LabPC 0.3531, Ubuntu 0.1930.

Coordinate changes alone do not prove semantic groups are meaningless: the encoder itself changes. They demonstrate that the old labels were not tied to persistent gradient-demand contexts.
The bounded retrospective audit used two independent native B2000/L2000/H30 gradient minibatches at each stage for both18000-update checkpoints, with no optimizer updates. Same-cohort cross-replicate gradient cosines ranged from -0.6262 to 0.8224. This is descriptive evidence of weak or inconsistent direction reproducibility, not a population CI from two repetitions.
The old procedure trained jointly throughout; there was no separate final joint-fine-tuning phase. Destruction of previously validated specialists is NOT established.

At exactly18000updates, a new matched development comparison(N512,L262144,seed98102605) gave seed10110.493405±.076276 versus seed10210.589358±.082048.
Paired seed101−seed102=-0.095953±0.093554;95%CI[-0.2793188016311673, 0.08741337262392485].
This does not isolate hardware, random seed, optimizer trajectory or extra final updates.

## Bottleneck classification

A: Confirmed proxy-based, changing grouping without persistent gradient-demand identities; usefulness not established.
B: Gradient conflicts occur; their reliable context structure is weak in the bounded replicate audit. A causal optimization benefit of projection is unsupported.
C: Router-context co-adaptation was indirect only. LabPC had a positive development routing intervention advantage; Ubuntu did not. No reproducible context competence reversal was established.
D: No evidence supports a separate destructive final-fine-tuning phase; no freeze/unfreeze redesign is justified.
E: No numerical failure established in completed v1 runs; high clipping is not itself a bug. The new v2 implementation had two separately documented GPU implementation bugs, corrected and retested.
F: Finite-contrast saturation and optimization limits remain plausible. The2k-contrast ceiling is near logged terminal training information, but this does not prove a larger contrast count would improve official EIG.

All arrays, source campaign paths, development curves and statistics are retained in fh_moe_v2_evidence.
