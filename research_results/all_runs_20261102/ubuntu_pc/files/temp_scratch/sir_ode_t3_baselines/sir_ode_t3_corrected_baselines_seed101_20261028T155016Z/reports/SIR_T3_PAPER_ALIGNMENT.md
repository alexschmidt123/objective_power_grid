# Pinned reference alignment

[Pinned DAD](https://github.com/ae-foster/dad/tree/4b1008174e1531d1f14601d83cef481c0f586f36): original full-horizon contrastive objective, set/sum encoder and native Adam recipe, adapted to normalized SIR time/count histories. Optimized Fixed uses the same finite-horizon objective with a nonadaptive trainable schedule; shared contrast reuse is a declared stochastic-minibatch adaptation.

[Pinned RL-BOED reference](https://github.com/csiro-mlai/RL-BOED/tree/54fdc7b6d23849ab1df8fd85d42af7c8961f930e), Garage2c5410e65680baf2d62d53eea42cfc4344e090a7. Native class bodies/update kernels are loaded from vendored originals. SIR adapter changes observed inputs, simulator and feasible continuous action only; replay/warmup/REDQ optimization retained. This is not historical REINFORCE.

[Pinned vsOED](https://github.com/wgshen/vsOED/tree/d0884f9ba30fbddc56618841c68af2f3bfe16539): native PGvsOED actor/critic/GMM posterior optimization. Published SIR used another physical implementation/data bank; current deterministic ODE is explicitly SIR_ADAPTED. An isolated invertible input-unit adapter scales observed time percentages/observations and current critic design; posterior density remains in native log-rate coordinates. No hidden-state inputs or vsOED imitation elsewhere.

[Pinned ScoreBED](https://github.com/angusphillips/Score_Matching_BED/tree/cc8e996e3ee3237ab4c9bd3ef80e2e4b4aa9cee4): original conservative score loss, Transformer and GRU policy kernels; scalar-potential hidden width256 follows the provided nondegenerate potential-head option. Required native dynamical schedule is50000scoresteps +500policysteps per restart,10restarts. The old wrapper forces one restart; this campaign DOES NOT reuse that primary-training shortcut. If full score fitting is infeasible, preserve profiling and block primary execution.

PAPER_ALIGNED describes verified native components. SIR_ADAPTED describes domain changes. COMPUTE_LIMITED describes capped or incomplete training, not paper-equivalent convergence. Exact upstream URLs/commits/hashes are in manifests/pinned_reference_verification.json; missing non-executable upstream files are recorded, not invented.
