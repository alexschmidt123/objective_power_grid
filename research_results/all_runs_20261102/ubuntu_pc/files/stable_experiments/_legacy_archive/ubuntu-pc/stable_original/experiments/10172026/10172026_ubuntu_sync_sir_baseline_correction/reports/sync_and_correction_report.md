# UbuntuPC sync and SIR baseline correction

GitHub main fast-forwarded from d09720e to 6a592c8. Pre-sync local SIR files, git diff and hashes are saved in pre_sync_source/ and metadata/. The git stash is retained. Applying it reported that files newly tracked by GitHub already existed: numerical solver and runtime files were independently checked against the backup and match; the changed campaign preparer remains available in the backup. No prior campaign checkpoint or result was overwritten.

The five current LabPC routing-extension source files were copied separately and their resulting content is frozen. They are local extensions beyond GitHub6a592c8, not represented as a new GitHub commit. The original relaxed-routing defaults are preserved. One obsolete test asserting ST was forbidden for switching variants was updated to reflect the already validated extension.

## Corrections and evidence

- ScoreBED SIR inherited a conservative-potential LayerNorm over two outputs. The pinned implementation explicitly documents gradient degeneracy for this scalar-design/scalar-observation case. The corrected run enables its existing potential_hidden_dim=128 option, used by the native gravimetry preset. Native transformer, score matching, policy-gradient estimator, policy backbone, physical likelihood and original10014/692990/296996 episode allocation remain unchanged. This dimension-specific native-head adaptation is disclosed; it is not claimed to be an exact paper reproduction. Smoke score-gradient norm123.0976 versus historical median7.65e-10 demonstrates restored gradient flow, not performance improvement.
- vsOED trained actor gives boundary outputs and zero input gradients on all1024 probe histories. At identical initialization, the raw-input probe puts38.1percent within0.01 of a percentage boundary; scaling removes this probe saturation. The correction converts design percentages /100 and observed counts /500 at native actor, critic and posterior feature inputs. It leaves physical observations, likelihood, exploration, reward, native networks and optimizers unchanged. This is a controlled input-scaling experiment; its sufficiency to prevent collapse is not yet established.

## Verification and execution

64 focused regression tests passed;3 local-fixture tests skipped. Native vsOED and ScoreBED launcher smokes passed, including feasible sequential times, finite computations and common evaluation. Source/config snapshots and SHA256 manifest are frozen before full training. Historical checkpoints are hashed under metadata/original_checkpoints.json. Both corrected jobs use run.sh through tools.journal_campaign.run_phase, one seed101 and exactly1M primary episodes each, final-budget selection. GPU jobs execute sequentially. Development evaluation remains seed1001 with2048 systems/L16384. No final-test seed is accessed and no other methods are retrained. Failed/partial runs are preserved and not automatically replaced.

Current status is recorded in metadata/status.json; native phase commands, launcher logs, exit status and completion manifests are in sir/seed101/. Results and paired trajectory-level Monte Carlo intervals will be generated from saved numerical outputs. A single training seed cannot establish stability or superiority.

LabPC active location correction queue was not modified or interrupted.
