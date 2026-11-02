# IEEE9 conference campaign — Ubuntu

Signed endpoint RoCoF, 3.5-second windows; T=3/4/5. Online simulation only.

| Setting | Value |
|---|---|
| Methods now | DAD, RL-sBOED, Myopic, Fixed, Random |
| Deferred | Step-DAD, matching DAD checkpoints |
| Training seeds | 101 reused; 202 and 303 fresh |
| Evaluation seeds | 1001, 1002, 1003 |
| Systems per evaluation seed | 512 |
| Updates / batch / learning rate | 2000 / 32 / 0.001 |
| Contrasts / planner particles / validation systems | 128 / 128 / 128 |
| Validation interval | 100 updates |

Seed 1001 was used in development: this is not an entirely untouched confirmation set.
Original 128-system results remain historical; all campaign scores use 512 systems.
Myopic/Random are training-independent and run once per horizon. Do not count their reused rows as independent trainings.

## Files

- `plan.json`: exact run commands and settings.
- `queue.json`: launched PIDs; one GPU job at a time using the existing queue script.
- `logs/`: one file per job.
- `T3/`, `T4/`, `T5/`: fresh runs and evaluation-only outputs.
- `checkpoint_transfer_receipt.json`: byte checks for transferred seed-101 records.
- `checkpoint_load_checks.json`: all nine learned checkpoints loaded strictly; source and settings verified.
- `source/`, `source_manifest.json`, `compatibility.json`: frozen historical scientific definitions with current CLI orchestration.
- `deferred_stepdad.json`: ready commands, NOT launched. Source the environment script before running them.

Each completed run must contain `completion.json` and `exit_code` equal to zero. Queue progression alone does not establish success.
Step-DAD uses 16 online refinement updates. No offline DAD retraining is needed once the matching training run completes.
