# Ten-method location comparison:1M total training episodes

Development results only. No DAD-MLP baseline. No five-million vsOED or two-million residual result is included.

| Method | Held-out EIG | Status |
|---|---:|---|
| Random | 8.26624 | historical_development |
| Optimized Fixed | 8.85782 | historical_development |
| DAD | 9.76665 | historical_development |
| RL-sBOED | 9.49698 | historical_development |
| vsOED | 8.58218 | historical_development |
| ScoreBED | 9.91568 | historical_development |
| MoE-sBOED | 9.82944 | historical_development |
| ResMoE-sBOED | 10.56342 | new_1M_total_development |
| SwitchMoE-sBOED | 10.23554 | historical_development |
| SwitchResMoE-sBOED | 10.52865 | new_1M_total_development |

All methods use common2000/L1million development evaluation1001. New residual methods include500k base+500k residual. Historical references remain single-seed final-checkpoint evidence. Across-seed final replicated results are not yet available. Machine-readable paired differences, simulation/likelihood costs and expert/switching diagnostics are in report.json.
