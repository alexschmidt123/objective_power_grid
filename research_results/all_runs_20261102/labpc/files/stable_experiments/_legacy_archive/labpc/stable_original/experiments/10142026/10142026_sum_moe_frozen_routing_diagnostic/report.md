# Frozen routing diagnostic

Dominant expert: 2 (zero-based; first measurement excluded).

| Policy | EIG |
|---|---:|
| Learned DSelect-1 | 9.38016055 |
| Fixed expert 2 | 4.72379565 |

Delta_route = 4.65636491 nats; paired SE 0.05586572; approximate95% interval [4.54686809, 4.76586172].

| Expert | Held-out selection | Training nonzero-gradient updates | Final20% nonzero-gradient updates |
|---|---:|---:|---:|
| 0 | 0.0000% | 13990/15625 | 1499/3125 |
| 1 | 28.5966% | 15625/15625 | 3125/3125 |
| 2 | 50.2241% | 15625/15625 | 3125/3125 |
| 3 | 21.1793% | 15485/15625 | 3015/3125 |

Non-negligible gradient threshold: post-clipping expert norm >1e-8. These counts are gradient proxies, not actual Adam weight-update counts.

| Measurement stage | Expert0 | Expert1 | Expert2 | Expert3 |
|---|---:|---:|---:|---:|
| 2 | 0.00% | 100.00% | 0.00% | 0.00% |
| 3 | 0.00% | 100.00% | 0.00% | 0.00% |
| 4 | 0.00% | 100.00% | 0.00% | 0.00% |
| 5 | 0.00% | 100.00% | 0.00% | 0.00% |
| 6 | 0.00% | 100.00% | 0.00% | 0.00% |
| 7 | 0.00% | 100.00% | 0.00% | 0.00% |
| 8 | 0.00% | 100.00% | 0.00% | 0.00% |
| 9 | 0.00% | 75.75% | 0.00% | 24.25% |
| 10 | 0.00% | 42.40% | 0.05% | 57.55% |
| 11 | 0.00% | 11.10% | 9.35% | 79.55% |
| 12 | 0.00% | 0.05% | 29.20% | 70.75% |
| 13 | 0.00% | 0.00% | 48.75% | 51.25% |
| 14 | 0.00% | 0.00% | 65.05% | 34.95% |
| 15 | 0.00% | 0.00% | 77.45% | 22.55% |
| 16 | 0.00% | 0.00% | 84.80% | 15.20% |
| 17 | 0.00% | 0.00% | 88.15% | 11.85% |
| 18 | 0.00% | 0.00% | 89.15% | 10.85% |
| 19 | 0.00% | 0.00% | 88.25% | 11.75% |
| 20 | 0.00% | 0.00% | 85.40% | 14.60% |
| 21 | 0.00% | 0.00% | 84.25% | 15.75% |
| 22 | 0.00% | 0.00% | 82.50% | 17.50% |
| 23 | 0.00% | 0.00% | 79.05% | 20.95% |
| 24 | 0.00% | 0.00% | 79.55% | 20.45% |
| 25 | 0.00% | 0.00% | 77.75% | 22.25% |
| 26 | 0.00% | 0.00% | 76.95% | 23.05% |
| 27 | 0.00% | 0.00% | 77.05% | 22.95% |
| 28 | 0.00% | 0.00% | 78.05% | 21.95% |
| 29 | 0.00% | 0.00% | 77.05% | 22.95% |
| 30 | 0.00% | 0.00% | 78.70% | 21.30% |

Learned routing clearly exceeds the frozen dominant-expert intervention on these paired cases.

Interval conditions on one trained checkpoint, the selected dominant expert and one shared contrast bank; it is not training-seed or contrast-bank uncertainty.
Dominant expert selected by usage on these same held-out/development cases, not by maximizing evaluated EIG.
A poor fixed-expert result does not prove beneficial specialization: forced routing induces different, potentially unfamiliar histories.
Evaluation1001 has already informed development; this is not untouched test evidence.
MSE is not recomputed; this is the requested EIG-only intervention.

Exactly one recommended next experiment: Test a matched193-input sum-single control, retaining the same encoder, stage feature, first-action treatment, head widths, optimizer and budget.

No follow-up implemented. All source artifact hashes and model weights verified unchanged.
