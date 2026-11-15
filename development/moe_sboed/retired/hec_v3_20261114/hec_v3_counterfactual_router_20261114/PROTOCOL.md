# Frozen-expert V3 paired counterfactual router pilot

This changes routing training only. Four trained V3 query pathways, standalone action heads, shared context, posterior networks, native critic and physical simulator remain frozen. Architecture and categorical execution remain unchanged. Training uses the original terminal-minus-prefix variational posterior reward. Final evaluation uses independent prior-sPCE, not the training surrogate.

## Two arms

Both start from the exact completed V3 selector runtime. Both run three rounds with 96 newly generated mixture-policy histories per round, one independently simulated trajectory per history. Stages are stratified over 0--29. Histories are collected under that arm's current router; continuation is held fixed throughout a round.

Each history gets 192 conditionally sampled latent draws. Every draw evaluates all four first actions with paired future noise and routing uniforms. All subsequent designs depend on branch-local observed histories. Sixty-four suffix draws supply training estimates; 128 independent draws supply diagnostic return measurements. Their finite posterior particle approximation is shared, a limitation rather than an exact posterior claim. Quality requires ESS >=256 and maximum normalized weight <=0.05; otherwise the run stops. Particle counts try 2,097,152 then 8,388,608, with chunked fallbacks of33,554,432 and134,217,728. Chunking retains the same proposal and exact likelihood with bounded GPU allocation. Stage zero uses exact normal-prior sampling.

The counterfactual arm fits a three-member value-advantage ensemble: 648 inputs (history embedding128, legacy256, responses256, proposed designs8), hidden128 SiLU, hidden64 SiLU, four outputs. Features are normalized using only the first64 histories. All64 are training histories, next16 validation histories for early stopping, last16 audit histories. Each member has300 AdamW updates, LR0.001, weight decay0.001; the best validation step is restored. Training targets are paired mean returns centered across experts. No expert or actor input includes latent truth.

The control arm retains the original frozen native critic to supply router targets. It collects exactly the same number of suffixes on its own histories, fits the same ensemble for diagnostics, but uses original critic advantages for updates. Thus both simulation expenditure and value-fit expenditure match. This is a frozen-critic credit-source control, **not an ordinary native actor/critic continuation baseline**. Additional suffixes have diagnostic rather than training value for its router.

Each router gets at most50 full-batch Adam steps per round, LR0.0001, finite-gradient checks and norm clipping1. Loss is negative expected estimated advantage plus KL(old router || new router). A step exceeding mean KL0.01 is reverted and the round ends; actual accepted updates are reported. Value regression and KL are estimator/trust-region mechanisms; the native information reward is unchanged. Subtracting a baseline alone is tested to leave gradients unchanged; the actual intervention is replacement of critic values by paired suffix estimates.

## Evaluation and limits

Untouched evaluation bank: seed202611145901,1024 trajectories,16 million prior contrasts from seed202611145902. Compare counterfactual router, critic-source control, original V3, uniform, all four standalone experts, original V2. Paired trajectory confidence intervals and Holm corrections are reported. No hyperparameter selection uses this bank. Historical external baselines are not rerun because this is a targeted routing diagnostic. Passing all internal comparisons including V2 is an ambitious pilot acceptance test, not evidence of broad ICML-level superiority.

Simulation observations are counted exactly, including all four suffix branches and history generation; posterior likelihood evaluations and EIG likelihood evaluations are separate compute costs. Arm counts must match exactly. Only one training lineage is tested; multi-seed confirmation and more heldout histories would be needed before a performance claim. Sixteen audit histories per round are exploratory and cannot settle small conditional gains. No online/deployment optimization is added.

## Preservation and execution

Raw data, models, scores and manifests stay in a new LabPC scratch directory. Git contains code and small reports only. Historical V2/V3 actor/runtime hashes and frozen context/expert digests must remain unchanged. GPU execution requires the existing idle-GPU lease; unrelated jobs are never interrupted. A two-hour wall cap bounds the pilot. Smoke validation uses4 histories,8 suffix draws,3 router steps and8 evaluation cases/4096 contrasts, solely for correctness and not scientific inference.
