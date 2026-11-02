# Recovery

No failed worker automatically restarts. Full policy/optimizer/scheduler/RNG checkpoints and the aggregate ledger persist. An interrupted active ledger blocks later stages pending review. Do not reset its spent counter or grant another allocation. Completed original/phase records are preserved. Source/config hashes must be verified before checkpoint restoration.
