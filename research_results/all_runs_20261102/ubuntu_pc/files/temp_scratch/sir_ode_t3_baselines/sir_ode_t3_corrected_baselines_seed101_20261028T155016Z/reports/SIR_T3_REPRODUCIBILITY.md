# Reproducibility and recovery

Training seed101. Fresh T3 model initialization, no T5 reuse. Development2026102831, resolution2026102832, confirmation2026102833, reserved restart-selection2026102834. Numerics2026102835, profiling2026102836, integration2026102837. Textual prior-use audit records scope/limitations; binary archives are not exhaustively inspected. Final confirmation is generated only AFTER final checkpoint/resolution freezing and is never consumed for selection.

Complete effective config is configs/sir_ode_t3_continuous_v1.yaml. Source files/config hashes, verified source archive, dirty Git revision metadata, Torch/JAX environments and device information are in manifests/. Ubuntu makes no Git commit/push. No STABLE result registration or LabPC access.

run.sh --sir-t3-campaign is the root workflow. Persistent sequential controller runs only one GPU child at once. Each child acquires the existing cache/device lock plus TEMP/STABLE .gpu0.lock and rejects existing foreign GPU processes. Interrupted/failed methods NEVER restart from zero automatically. --continue skips complete/blocked methods and refuses an existing partial method pending explicit recovery review; it continues only unstarted work. Native runtime snapshots include replay/critics/targets/optimizers where applicable; policy snapshots include optimizer/scheduler/RNG/counters. They support manual state recovery but this campaign does not claim an automatically tested exact native resume path. Prior charged work is never silently zeroed. A child wall watchdog affects only its own process group.

Training initialization, CUDA preparation, simulator/likelihood work, all native trainable stages, replay/auxiliary updates, checkpoint serialization and hashing are charged. Development/final evaluation intervals are separately timed. Latest fully serialized budget-eligible checkpoint is selected, never maximum test/development EIG. The runner reserves time for final serialization; selected checkpoint cost sidecars include actual completed work. Native setup and normalization would be charged if ScoreBED became eligible; profiling costs remain separate and no profiled weights are reused.


Observed 2026-10-28T16:30:11.094030+00:00: fixed RUNNING; 900 updates, 1800000 fresh episodes, 131.566 charged seconds. ControllerPID 1289522,workerPID 1289602. This snapshot is not completion.
