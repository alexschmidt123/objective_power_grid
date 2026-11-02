## Frozen model and completed run

The model has one always-active shared width8 SwiGLU expert, seven routed width8 experts, top-three routed activation, a shared 3-256-ReLU-16 sum encoder, and one shared design head. Experts emit hidden features, not independent experiment-design policies. There are 8,434 parameters, with 6,898 activated per context. The latent output is mapped to the original [-4,4] box.

Training seed101 completed 14232 updates in 14390.523 charged seconds and stopped at the compute cap. The selected checkpoint is update14232. The latest logged sPCE is 8.749205 at update14200; no unlogged value at the final update is invented. Training used 8,192 contrasts and 30 stages.

Checkpoint SHA256: f16860948e467bec75fa3ae07938937786322347e802caa31c24c65b0bc05a91.

The legacy controller says FAILED_PRESERVED because the report writer attempted Path-plus-string concatenation AFTER training, both IR evaluations, and comparisons had completed. The checkpoint is numerically finite, all frozen source/config hashes were verified, and saved per-trajectory scores reproduce the official summary. This report repairs artifact reporting in a separate worktree; it does not rewrite the original registry or results.
