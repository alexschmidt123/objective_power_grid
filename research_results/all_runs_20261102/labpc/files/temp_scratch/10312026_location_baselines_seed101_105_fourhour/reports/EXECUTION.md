# Execution and recovery

Four seed101 training artifacts are independently hash-verified copies. The originals remain untouched. RLseed101 is rerun for conservative strict startup/serialization accounting. New seeds102–105 start fresh. Three established project device locks and a controller lock serialize GPU use. No unrelated process is preempted.

Native trainers reserve120seconds for finalization. An external process-group guard stops only its own worker before the14400-second charged limit if a native cutoff fails. Such a safety stop is incomplete, not a successful result. Completed runs are skipped on controller recovery; interrupted phases are never silently retrained or reset. Source/config drift fails closed. Preserve partial evidence; exact mid-phase resume is not implemented for every native trainer.

Results and statuses update after each completed method. Final confirmation arrays are opened only after training candidates freeze. No automatic STABLE promotion, GitHub push or merge.
