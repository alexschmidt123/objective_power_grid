# Replacement V3 validation

The complete original V2 bank/scorer check passed over2,000 cases/1M contrasts. Initial learned mixture score vector is byte-value identical to V2: mean12.929352605692825,maximum absolute score error0,maximum action error0. This validates preservation,not improvement.

Functional checks passed at B=2 for stages0,14,29: exact V2 first/response packets,gates,mixed actions,uniform actions and one-hot single pathways. Each single mode executes only its own current query; standalone exports contain one expert and common providers/decoder. Future-history NaNs are masked. Output/router layers have finite nonzero native-critic gradients at initialization. Zero-initialized output layers intentionally give zero hidden gradients initially; a throwaway nonzero-output perturbation verified hidden connectivity.

Registered actor parameters169,382; trainable private-innovation/router parameters65,476. Frozen model providers also store weights as buffers. New private networks have distinct13/28/20/26-dimensional semantic inputs and128-dimensional outputs; router correction input147.

Candidate and original-V2 control passed one genuine native update each:1 actor,5 critic,150 posterior updates. Frozen actor state and historical V2 checkpoint/runtime hashes were preserved. Neither smoke checkpoint initializes the pilot. The candidate smoke02 additionally verifies target/online state identity before its first update. Smoke01 used inherited target components and remains preserved; the final protocol initializes the new target from online weights so frozen projection/offset buffers cannot retain permanent stale target values. This target initialization differs from the control and is disclosed.

Nineteen failed-V3 source files were moved byte-identically to the retired versioned directory and additionally archived outside Git. Historical checkpoint paths/bytes and all results remain intact. No running jobs were interrupted. Read-only retired-reference loader verified all source hashes and successfully constructed/loaded the old actor and ran a finite bounded2D forward pass. The canonical evaluator now uses this loader rather than an active failed-V3 source directory.

All changed Python and JSON are compiled/parsed by the synchronization helper. The bounded pilot is only a short one-lineage diagnostic. It does not establish specialization, superiority over uniform/singles or ICML-level performance.
