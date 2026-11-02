# All-run result-file snapshot — both PCs

This directory preserves experiment evidence from **both LabPC and UbuntuPC**, across all benchmarks and training seeds found under STABLE experiments, TEMP scratch runs and project reports. It includes current runs, completed runs, negative results, interrupted runs and historical archives. **Archival inclusion is not scientific validation or promotion.** Original status/provenance is retained. The original experiments were neither restarted nor changed by this sync.

| Host | Original result-file paths indexed | Unique stored files | Snapshot UTC |
|---|---:|---:|---|
| [labpc](labpc/INDEX.csv) | 34,765 | 17,986 | 2026-11-02T20:39:49.531598+00:00 |
| [ubuntu_pc](ubuntu_pc/INDEX.csv) | 14,529 | 4,497 | 2026-11-02T20:35:01.883008+00:00 |

## Included

EIG/metric tables and score arrays; paired comparisons; training and evaluation logs; plots; reports; effective configurations; compute counters; run/status manifests; source/checkpoint hashes and scientific provenance. Supplementary text status/exit-code/checksum files and historical reference reports are also preserved. Large text results are **losslessly gzip-compressed**, not truncated or dropped. Large individual stored payloads are split into40MiB parts when needed for GitHub limits.

## Read and recover files

Each host's **INDEX.csv** maps `scope + original_path` to `published_path`, with original SHA256, size, capture information and encoding. Identical file contents are stored once per host; several original paths can map to the same canonical file. `files/` contains the actual result bytes. `SUMMARY.json` records totals. Paths are relative to the four source scopes described there.

To find one run, search INDEX.csv for its experiment identifier. Canonical files retain their original directory hierarchy whenever unique. A large CSV index can be downloaded from GitHub's Raw link.

To reconstruct selected original files after cloning this branch:

```sh
python3 research_results/all_runs_20261102/restore_results.py \
  research_results/all_runs_20261102/labpc /path/to/new/result-export \
  --match journal_LF_T30_native_final
```

Omit `--match` to reconstruct all indexed paths for that host. The tool decompresses/reassembles payloads, verifies SHA256, refuses overwrites, and never starts training.

## Exclusions and interpretation

Each host's **EXCLUDED.csv** lists omitted paths and reasons. Models/checkpoints, neural weights/optimizer state, raw simulation/contrast/noise banks and raw posterior/sample datasets stay on the PCs under the existing storage policy. Source/environment caches and unrelated non-result files are excluded. Small numerical EIG/diagnostic outputs are retained; no held-out final bank was evaluated or published as an input dataset. File symlinks inside the projects are read when safe; directory aliases are not recursively traversed because their canonical records are already inventoried.

These are **point-in-time snapshots**, not an assertion that ongoing jobs completed or a schedule to upload future outputs. Runtime statuses and counts mean what they meant at capture. The5-seed native LabPC campaign is ongoing; unfinished EIG values have not been fabricated. Legacy/excluded historical methods are not automatically inserted into current benchmark tables. Original byte hashes distinguish snapshots and duplicates.

Only LabPC committed and pushed. No main merge, STABLE promotion, dataset deletion or checkpoint deletion occurred.
