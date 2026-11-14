# Frozen two-failure diagnosis validation

No optimizer steps, retraining, checkpoint replacement, or native source edits occurred.

- Main audit completed in182.5 seconds on the exact original V2 bank/scorer. Three diagnostic joint-head assemblies were evaluated over all2000 cases and1M contrasts. Original-core bridge parity passed at B=2, stages0,14,29.
- All nine selected value models were reconstructed from their original seeded initializations and closed-form first AdamW updates. Maximum saved-weight reconstruction error was7.45058e-9. This validates the untrained-versus-selected value comparison without updating a historical model.
- Router return and KL gradients were measured at each saved endpoint on its original64 fitting histories. State digests remained unchanged; finite gradients and actual KL matched recorded update receipts.
- The fixed-router factorial test completed in54.7 seconds. Its inherited-router categorical score vector reproduced the previous saved vector with maximum error2.30926e-14. Router/decoder interaction was tested using paired cases; comparisons have Holm correction within their stated diagnostic families.
- The canonical reference run already reproduced all six V2/DAD/vsOED/RL-sBOED/ScoreBED reference vectors exactly. Its protocol and checkpoint certificate were checked before the new diagnostics. Existing score vectors and model/data files were hashed before and after; preservation checks passed.
- The initial factorial import collided with the preexisting native module named audit and stopped before creating an output directory or running simulations. diagnostic_common.py now loads this new audit under a unique module name; historical modules remain unchanged. The corrected factorial and report publisher completed successfully.

The bridge assemblies are mechanism tests, not new accepted standalone-expert methods. These results do not establish the absence of conditional routing headroom, intrinsic inferiority of decision-level MoE, a complete cause of the original native critic's allocation, or prospective EIG gains from the proposed corrections.

Analytic correction: the original audit's exponential-reweighting optimum applies to KL(new||old), while the actual update uses KL(old||new). reverse_kl.py corrects that calculation to p=old/(tau-A), checks normalization, KKT stationarity, the zero-advantage identity and objective improvement, and writes CORRECTED_RESULTS.json. The original raw result/source are preserved; measured return/KL gradients and bridge scores are unaffected. The published report uses the corrected artifact and never treats the old exponential formula as evidence.
