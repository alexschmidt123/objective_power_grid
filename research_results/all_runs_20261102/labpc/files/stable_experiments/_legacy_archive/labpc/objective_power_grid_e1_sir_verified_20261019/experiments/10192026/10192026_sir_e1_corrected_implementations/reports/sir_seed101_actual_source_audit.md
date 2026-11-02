# SIR actual-source audit

All ten runs used frozen commit23f629c, not GitHub main. Every source-file copy was verified against its own run manifest. Native RL used pinned REDQ, discount0.9, two critics/M2,64 updates per100 episodes; legacy research_redq was not selected. DAD checkpoint uses shared empty-history emitter revision1 and has no separately learned first_design parameter. All four family checkpoints specify straight_through; both switching checkpoints specify switching_training_revision2 and learned_switch. Both residual checkpoints contain the identical SIR-specific500k base hash, verified bitwise unchanged.

Full frozen configuration and optimizer records for location and SIR are in metadata/location_sir_effective_configurations.json. Native vendor hash comparisons are in the matrix. This verifies provenance, not automatically score-estimation accuracy or convergence. Original seed101 evaluations and checkpoints are preserved. Seeds102-105 remain held.
