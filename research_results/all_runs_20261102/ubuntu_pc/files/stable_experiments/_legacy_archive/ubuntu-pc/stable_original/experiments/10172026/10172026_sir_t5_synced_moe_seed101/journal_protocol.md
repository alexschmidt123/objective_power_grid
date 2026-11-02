# Synced SIR T=5 four-variant development rerun

Requested rerun of all four canonical MoE variants on UbuntuPC, training seed101. Use synced four-variant straight-through DSelect1 treatment consistently, preserving64/64tanh experts, sum encoder, original30percent/95percent gamma schedule, optimizer, regularizers, component clipping, first design, chronological action mapping, explicit SIR likelihood and full-horizon L1024sPCE. Switching gates stay learned; no gate schedule is added. Historical relaxed SIR runs remain separate and immutable.

MoE and SwitchMoE train1M episodes from scratch. ResMoE and SwitchResMoE reuse the identical compatible independently trained seed101SIR500kbase and train500knew residual episodes each, frozen base verified bitwise on completion. Each complete model accounts for1M primary episodes. Physical computation reuses the base; do not count its reuse as new work.

Final-budget selection. Common development EIG evaluation2048trajectories/L16384/seed1001, with diagnostic validation772001. This is not final confirmation and estimator precision still needs validation. No final evaluation seed is accessed. No additional training seeds or architecture search. Preserve negative results and single-expert collapse if observed.

Queue waits for the existing Ubuntu baseline-correction controller to reach a terminal state and exit, then verifies no GPU compute process is active before every phase. All training uses canonical run.sh through existing tools.journal_campaign.run_phase. No frozen source/configuration in the baseline campaign is changed. Source/config snapshots and hashes precede training. Failure/partial outputs are retained with explicit status; no automatic rerun or overwrite.
