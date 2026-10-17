# MoE-sBOED straight-through routing development experiment

Only training routing changed. Seeds103/105; exactly1M primary episodes each; final-budget checkpoint selection. Original runs, all other variants and final evaluation seed8831001 remain untouched. This is an optional MoE-sBOED training mode, not a new named variant.

## Final paired development EIG

2000 paired trajectories, L=1,000,000, seed1001. Native hard scores reused only after truth/noise/contrast-bank identity checks and independent16-case checkpoint score replay. Relaxed scores use the same draws and existing common scorer.

| Seed | Training routing | Hard EIG ± MC SE | Relaxed EIG | Relaxed − hard [95% MC interval] | Training seconds |
|---|---|---:|---:|---|---:|
| 103 | original | 6.25322 ± 0.05495 | 6.27053 | +0.01732 [-0.00059, +0.03522] | 1777.28 |
| 103 | straight_through | 10.79381 ± 0.04425 | 10.79381 | +0.00000 [+0.00000, +0.00000] | 1643.23 |
| 105 | original | 10.69964 ± 0.04637 | 10.69964 | +0.00000 [+0.00000, +0.00000] | 1562.78 |
| 105 | straight_through | 10.84271 ± 0.04703 | 10.84271 | +0.00000 [+0.00000, +0.00000] | 1647.61 |

| Seed | ST − original hard EIG | Paired95% MC interval |
|---|---:|---|
| 103 | +4.54059 | [+4.40759, +4.67360] |
| 105 | +0.14307 | [+0.03778, +0.24835] |

These intervals measure trajectory-level Monte Carlo uncertainty, not variability over independent training runs. Two selected development seeds cannot establish general superiority. All negative differences are retained.

## Late-stage behavior

### Seed103

All-checkpoint sustained-drop diagnostic: `{"first_saved_deterioration": null, "note": "No sustained drop satisfying the stated rule on saved checkpoints; not proof of general stability"}`. Late-only750k–1M diagnostic: `{"first_saved_deterioration": null, "note": "No sustained drop satisfying the stated rule on saved checkpoints; not proof of general stability"}`.

Final minus750016-episode hard development EIG: `{"difference": -0.04256179779766591, "mc_se": 0.08765887954123718, "ci95": [-0.21437320169849078, 0.12924960610315894]}`.

Final minus800000-episode hard development EIG: `{"difference": 0.12781216780331658, "mc_se": 0.0729000959465989, "ci95": [-0.01507202025201726, 0.27069635585865043]}`.

### Seed105

All-checkpoint sustained-drop diagnostic: `{"first_saved_deterioration": 825024, "anchor": 750016, "previous_saved_checkpoint": 812544, "three_paired_changes": [{"difference": -0.3529424936681532, "mc_se": 0.09269974334822585, "ci95": [-0.5346339906306758, -0.17125099670563054]}, {"difference": -0.549540078579985, "mc_se": 0.10423266796107035, "ci95": [-0.7538361077836828, -0.3452440493762871]}, {"difference": -0.3493296629595471, "mc_se": 0.09166527381924917, "ci95": [-0.5289935996452755, -0.1696657262738187]}]}`. Late-only750k–1M diagnostic: `{"first_saved_deterioration": 825024, "anchor": 750016, "previous_saved_checkpoint": 812544, "three_paired_changes": [{"difference": -0.3529424936681532, "mc_se": 0.09269974334822585, "ci95": [-0.5346339906306758, -0.17125099670563054]}, {"difference": -0.549540078579985, "mc_se": 0.10423266796107035, "ci95": [-0.7538361077836828, -0.3452440493762871]}, {"difference": -0.3493296629595471, "mc_se": 0.09166527381924917, "ci95": [-0.5289935996452755, -0.1696657262738187]}]}`.

Final minus750016-episode hard development EIG: `{"difference": -0.06292910559661283, "mc_se": 0.0901722196621454, "ci95": [-0.23966665613441782, 0.11380844494119216]}`.

Final minus800000-episode hard development EIG: `{"difference": 0.41510568577940177, "mc_se": 0.0931350710356701, "ci95": [0.23256094654948836, 0.5976504250093152]}`.

The onset rule is exploratory and not adjusted for repeated looks. Original105 has no saved intermediate model weights; its original hard validation curves are shown at their actual episode counts, and an intermediate relaxed comparison is unavailable. No original-setting training was repeated to fill this gap.

## Utilization and gradient exposure

| Seed | Window | E1/E2/E3/E4 hard selection | E1/E2/E3/E4 gradient exposure | Median selector saturation | Median router step |
|---|---|---|---|---:|---:|
| 103 | 0-200000 | 100.0%/0.0%/0.0%/0.0% | 100.0%/0.0%/0.0%/0.0% | 81.34% | 7.9163e-05 |
| 103 | 200000-500000 | 100.0%/0.0%/0.0%/0.0% | 100.0%/0.0%/0.0%/0.0% | 87.82% | 6.2321e-05 |
| 103 | 500000-750000 | 100.0%/0.0%/0.0%/0.0% | 100.0%/0.0%/0.0%/0.0% | 96.71% | 2.8387e-05 |
| 103 | 750000-800000 | 100.0%/0.0%/0.0%/0.0% | 100.0%/0.0%/0.0%/0.0% | 100.00% | 0 |
| 103 | 800000-850000 | 100.0%/0.0%/0.0%/0.0% | 100.0%/0.0%/0.0%/0.0% | 100.00% | 0 |
| 103 | 850000-900000 | 100.0%/0.0%/0.0%/0.0% | 100.0%/0.0%/0.0%/0.0% | 100.00% | 0 |
| 103 | 900000-1000001 | 100.0%/0.0%/0.0%/0.0% | 100.0%/0.0%/0.0%/0.0% | 100.00% | 0 |
| 105 | 0-200000 | 0.0%/0.0%/100.0%/0.0% | 0.0%/0.0%/100.0%/0.0% | 74.46% | 8.7318e-05 |
| 105 | 200000-500000 | 0.0%/0.0%/100.0%/0.0% | 0.0%/0.0%/100.0%/0.0% | 83.51% | 6.8091e-05 |
| 105 | 500000-750000 | 0.0%/0.0%/100.0%/0.0% | 0.0%/0.0%/100.0%/0.0% | 94.83% | 3.8905e-05 |
| 105 | 750000-800000 | 0.0%/0.0%/100.0%/0.0% | 0.0%/0.0%/100.0%/0.0% | 100.00% | 0 |
| 105 | 800000-850000 | 0.0%/0.0%/100.0%/0.0% | 0.0%/0.0%/100.0%/0.0% | 100.00% | 0 |
| 105 | 850000-900000 | 0.0%/0.0%/100.0%/0.0% | 0.0%/0.0%/100.0%/0.0% | 100.00% | 0 |
| 105 | 900000-1000001 | 0.0%/0.0%/100.0%/0.0% | 0.0%/0.0%/100.0%/0.0% | 100.00% | 0 |

Gradient exposure threshold is norm>1e-8. JSON records zero-gradient fractions, raw/clipped gradient distributions, actual parameter increments and selector-logit quantiles. Full stage-wise selection frequencies are saved separately. In hard-forward training all four proposals are computed for router backward gradients, but only selected experts receive direct action gradients. A stable high-EIG policy dominated by one expert does not establish beneficial expert specialization.

## Correctness and implementation

13 focused checks passed. Exact hard routing weights at gamma1/.1/.001, native zero-logit ties, exact fixed-context relaxed-router gradient equivalence, unchanged initial parameters, full hard-forward rollout within1e-12, unchanged relaxed regularizers, selected-expert gradient flow, exclusion of other variants, historical103/105 checkpoint prediction compatibility, one-expert hard inference, checkpoint roundtrip and finite full-batch/L1024 sPCE gradients. Measured rollout differences: actions8.88e-16, observations1.07e-14, latents3.05e-16 (dense versus sparse batched arithmetic).

The implementation uses w_ST=stop_gradient(w_hard)+(w_soft-stop_gradient(w_soft)); all four proposals enter the surrogate action expression. This is a biased routing-gradient surrogate, not a new EIG estimator. Native relaxed load/selector regularizers and all optimization settings are preserved. A selected-expert-only sparse dispatcher during training would incorrectly remove unselected proposal contributions to the routing gradient; it is deliberately not used here.

## Interpretation and adoption

Inspect both seed103 recovery and seed105 preservation; do not choose the favorable seed. If the joint results are promising, recommend an unchanged five-original-seed replication before changing the journal protocol. If one seed deteriorates or utilization/gradient exposure collapses, retain the negative result and do not claim the correction solved instability. An absence of a sustained late drop does not imply successful training if the policy was already poor. No further training or hyperparameter tuning is automatically launched.

## Accounting and reproduction

Two full runs use2,000,000 primary training episodes total,2,048,000,000 contrast trajectories, and61,500,000,000 truth-inclusive likelihood particle-stages. No pretraining. Diagnostic validation/evaluation and correctness generation are recorded separately. Original optimizer, learning-rate scheduler, regularization, clipping and gamma clock remain unchanged. Each run uses rootrun.sh; exact arguments are in `commands/seed103.json` and `commands/seed105.json`. Protocol, source hashes, original checkpoint hashes and patch are at the campaign root. Final checkpoint paths/hashes and wall times are in `completion/` and `final_eig.json`.

Source changes: `src/research/sparse_moe.py`, `src/research/moe_family.py`, `src/research/family_training.py`, plus configurable diagnostic checkpoint observation in `src/research/moe_diagnostic_logging.py`. Historical source copies remain unchanged.

![Development diagnostics](development_diagnostics.png)

Machine-readable outputs: `results.csv`, `paired_comparisons.csv`, `final_eig.json`, `development_curves.json`, `diagnostics.json`, and stage-wise utilization JSONs. Saved hard/relaxed per-trajectory score arrays support paired uncertainty reconstruction.

## Observed outcome, without checkpoint replacement

Seed103: final hard EIG change+4.54059, paired95% interval[4.407585076035269, 4.673603695531175]; final-minus-pre800k change-0.04256, interval[-0.21437320169849078, 0.12924960610315894]; late sustained-drop onset=None. Final hard expert frequencies=[1.0, 0.0, 0.0, 0.0].
At least99% of final decisions use one expert. This is expert collapse/underutilization, not evidence of four useful specialized policies.
Seed105: final hard EIG change+0.14307, paired95% interval[0.03778261334988943, 0.24835312529377623]; final-minus-pre800k change-0.06293, interval[-0.23966665613441782, 0.11380844494119216]; late sustained-drop onset=825024. Final hard expert frequencies=[0.0, 0.0, 1.0, 0.0].
At least99% of final decisions use one expert. This is expert collapse/underutilization, not evidence of four useful specialized policies.
The seed103 recovery is supported conditionally on these development cases, with no clear negative seed105 contrast at the nominal Monte Carlo level. This merits an unchanged five-seed development replication before adoption; it does not establish noninferiority, seed-level robustness or beneficial specialization.
