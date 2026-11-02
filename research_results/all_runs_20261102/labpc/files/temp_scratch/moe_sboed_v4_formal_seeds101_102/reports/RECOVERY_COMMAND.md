# Reviewed recovery only

Do not invoke while another formal controller runs. Inspect failure and latest budget-eligible checkpoint, then use root-sourced environment: python3 -m src.research.v4_formal_seeds resume SEED /absolute/checkpoint.pt. Controller acquires normal GPU lease and preserves prior charged compute and failed evidence, restores optimizer/scheduler/RNG, then continues remaining seeds. No automatic failed-run restart.
