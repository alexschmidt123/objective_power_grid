# RR correctness evidence

Scientific training is gated on successful implementation,physics,preparation
reload and evaluation receipts,plus compatible baseline preflights. They verify:

1. Bitwise initial uniform-HEC action identity for all30 stages and all6 pairs.
2. Exact unordered Plackett–Luce pair probabilities,normalization,extreme-logit
   stability,finite differences,score-function identity and sampling frequencies.
3. Zero-head routing/adaptation gradient block and nonzero-head gradient paths.
4. No independent history/bias correction bypass; actual selection changes design.
5. Future-mutation invariance and native posterior-prefix construction.
6. Explicit native likelihood and gamma1 variational reward telescoping.
7. Bounds,conditional prefix resume and changed-policy whole-trajectory simulation.
8. Full10000-episode native replay batch:300000 actor rows,1actor/5critic/
   150posterior updates,1000new trajectories/30000observations.
9. Identical original/target frozen buffers after native Polyak and Adam.
10. Native runtime roundtrip and identical prepared weights,critic,Adam/LR into
    both arms; real EIG preparation opens routing and adaptation gradients.
11. Independent literal prior-contrast scoring,positive included exactly once.
12. Original checkpoint hashes unchanged and the main EIG bank never accessed.

A preparation-loader validation failure exposed CPU RNG ByteTensor states being
incorrectly moved to CUDA. The loader now remaps only originally CUDA storages;
CPU RNG states stay on CPU. Training stays gated until reload checks pass.
An evaluator fixture dtype mismatch was addressed with lossless reference
promotion,retaining exact zero-tolerance physical trajectory assertions.
No correctness tolerance was relaxed to start a run. Disposable validation
weights never initialize the scientific arms; their compute is separately counted.
