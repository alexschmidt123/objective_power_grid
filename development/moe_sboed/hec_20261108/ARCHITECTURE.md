# Heterogeneous model queries and collaborative residual design

The frozen successful SG policy proposes a0(h). Its stage-dependent variational Gaussian-mixture posterior and native critic are reused as frozen model-information providers. They are stored as buffers so native target Polyak updates cannot alter them. This is amortized approximate inference, not an independent posterior solver or exact conditional-EIG oracle.

A trainable 117→128→128 history encoder is copied from SG. Its compatible Adam moments are retained. A 144→64→20 commander receives that encoding, the provisional design and a 14-coordinate belief summary (mean, log-variance and six correlations). Commands split into dimensions 4,6,6,4 and control specialist computations before fusion.

Expert 1 uses a 9→64→64 mixture-token network and command-dependent region pooling. Expert 2 uses a 4→32→64 likelihood-token network with command-dependent designs and hypothesis weights. Expert 3 maps 14 Fisher/direction coordinates through 96→64. Expert 4 maps 14 critic/predictive/horizon/command coordinates through 64→32→64. Distinct readouts have 14,9,9,3 outputs and are used only in preparation and diagnostics.

The candidate coordinator sees the history and all four feature vectors (384→64→4). The control has the same modules and tensor initialization but uses fixed zero commands and a history-only 128→4 soft router. Both retain every expert pathway and use weighted feature concatenation, never physical-action averaging. A 384→128→2 fusion produces a bounded correction to a0. Its last layer starts at zero, preserving the SG action exactly.

All new shape-incompatible expert modules start fresh. No inherited expert weights or Adam states are claimed for them. The new modules are prepared once, then copied identically to both arms. Inherited critic/posterior optimizers, replay and schedules continue. Additional effective commander/coordination capacity is an unavoidable control limitation, disclosed rather than hidden.

Large native 10,000×30 actor batches use exact gradient checkpointing in 4,096-row chunks; frozen belief parameters are computed stage-wise first. This changes memory use, not the objective or minibatch.
