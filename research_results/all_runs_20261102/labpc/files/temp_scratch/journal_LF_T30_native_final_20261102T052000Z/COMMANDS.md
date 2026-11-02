# Inspection and control

Status: `cat /home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/journal_LF_T30_native_final_20261102T052000Z/status.json`
Queue: `cat /home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/journal_LF_T30_native_final_20261102T052000Z/production_manifest.csv`
Safe boundary stop: `touch /home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/journal_LF_T30_native_final_20261102T052000Z/CANCEL` (finishes active method/evaluation; launches no next job).
Do not restart manager over an incomplete attempt; no exact-resume claim.
Later one-shot final inference, ONLY after the user freezes MoE: `/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python /home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/journal_LF_T30_native_final_20261102T052000Z/source/final_once.py --user-moe-freeze /path/to/user_approved_moe_freeze.json`.
Receipt must name frozen MoE checkpoint+SHA256, hyperparameters_frozen=true and user_approved_final_evaluation=true. Manager never calls this command. Final bank not transferred to Ubuntu.
