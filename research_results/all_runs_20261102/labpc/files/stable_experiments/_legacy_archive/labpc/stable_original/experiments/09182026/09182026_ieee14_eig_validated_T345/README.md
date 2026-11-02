# IEEE14 EIG — T=3/4/5, all seven methods

LabPC RTX4090; train seed 101, evaluation seed 1001.
Signed endpoint RoCoF, 3.5-second windows, 128 systems, 2000 updates,
batch32, learning rate0.001, 128 contrasts/planner particles/validation systems.
DAD and direct MoE share validation-selected fresh Fixed initialization.
Step-DAD uses the matching DAD with 16 online refinement updates.
The model has five retained machines, ten uncertain M/K parameters,
carried states and increasing continuous durations. All data are simulated.

- `validation/`: regression, seven-method smoke, sensitivity and timing receipts.
- `source/` and `source_manifest.json`: frozen executable source.
- `plan.json`: complete commands/settings.
- `queue.json` and `logs/`: submitted PIDs and progress.
- `T3/run`, `T4/run`, `T5/run`: results and model checkpoints.

A run is complete only when its completion.json and zero exit_code exist.
Single train/evaluation seed results cannot establish across-seed superiority.
