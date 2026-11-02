# Safe handoff

{
  "marker_preexisted": false,
  "dad_cancel_preexisted": false,
  "dad_status_before": {
    "status": "WAITING_GPU",
    "queue_pid": 1217692,
    "existing_controller_unmodified": true,
    "allocation_note": "Existing controller holds locks across entire queue; not necessarily available after RL worker."
  }
}

Created /home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/journal_location_T30_2M_seed101_105_20261101T172522Z/STOP_AFTER_JOB using the existing supported boundary stop. Active worker finishes training and its evaluation. All scientific manifests/results/checkpoints untouched. Superseded ONLY the unstarted DAD waiting dispatcher via its documented CANCEL marker to avoid a second claimant; its source/audit/specification remain preserved. The consolidated primary inventory will reuse valid native seed101 DAD rather than retrain it.

Restore only on explicit request: use `python /home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/journal_location_T30_labpc_native4h_complete_20261101T194734Z/source/restore_2m.py --dry-run` to inspect remaining queued jobs; `--execute` restores only originally QUEUED, never-partially-started jobs in a new coordination folder with exclusive GPU locks. Do NOT restart old supervisor from the beginning. Restoration refuses while formal coordinator is alive. The former standalone DAD queue can be reinstated only if scientifically needed by removing its CANCEL marker after dispatcher exit and invoking its archived exact queue command; doing so while formal queue owns GPU is unnecessary.
