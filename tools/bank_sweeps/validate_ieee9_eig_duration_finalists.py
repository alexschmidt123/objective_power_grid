#!/usr/bin/env python3
"""High-accuracy common-random-number validation of full-grid finalists."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layout import make_experiment_dir_name
from tools.bank_sweeps.search_ieee9_eig_duration_sets_full import exact_audit
from tools.bank_sweeps.sweep_ieee9_eig_duration_sets import (
    load_catalog, load_support_centres, resolve_pool_actions, write_csv,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--bank-dir", default="data/ieee9_duration_dense_0p01")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--support-size", type=int, default=384)
    p.add_argument("--outer-fantasies", type=int, default=64)
    p.add_argument("--inner-fantasies", type=int, default=32)
    p.add_argument("--seeds", default="101,202,303,404,505,606,707,808,909,1010")
    p.add_argument("--N-obs", dest="n_obs", type=int, default=5)
    p.add_argument("--noise-sigma", type=float, default=0.01)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()

    source = (ROOT / args.source).resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))[: args.top]
    if not source_rows:
        raise SystemExit(f"no finalists in {source}")
    duration_sets = [
        tuple(float(x) for x in row["durations_s"].split(";"))
        for row in source_rows
    ]
    full_durations = tuple(round(0.20 + 0.01 * i, 2) for i in range(281))
    duration_index = {d: i for i, d in enumerate(full_durations)}
    keys = [tuple(duration_index[round(d, 2)] for d in values) for values in duration_sets]
    bank_dir = (ROOT / args.bank_dir).resolve()
    catalog = load_catalog(bank_dir)
    bank_ids, _ = resolve_pool_actions(catalog, full_durations)
    centres, obs_indices = load_support_centres(
        bank_dir, bank_ids, n_obs=args.n_obs,
        support_size=args.support_size, support_seed=7919,
    )
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    folder = make_experiment_dir_name(
        "ieee9_duration_finalist_validation", "eig_based", 2,
        n_obs=args.n_obs, noise_sigma=args.noise_sigma,
    )
    out = ROOT / "experiments" / folder
    diag = out / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = {
        "system": "ieee9", "study_type": "high_accuracy_duration_finalist_validation",
        "source": str(source.relative_to(ROOT)), "n_candidates": len(keys),
        "duration_grid_s": [0.20, 3.00, 0.01],
        "combination_space": math.comb(281, 6), "support_size": args.support_size,
        "outer_fantasies": args.outer_fantasies,
        "inner_fantasies": args.inner_fantasies, "seeds": list(seeds),
        "common_random_numbers": True, "N_obs": args.n_obs,
        "noise_sigma": args.noise_sigma, "status": "running",
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    for i, key in enumerate(keys, 1):
        row = exact_audit(
            key, durations=full_durations, all_centres=centres,
            n_buses=len(catalog.buses), sigma=args.noise_sigma, seeds=seeds,
            outer=args.outer_fantasies, inner=args.inner_fantasies,
            device=device, min_branch_share=0.10, max_dominance=0.80,
        )
        # The decision is based on effect-size uncertainty and observed duration
        # branching. A per-seed first-action mismatch is redundant and brittle.
        passed = (
            float(row["adaptive_gap_lcb95"]) > 0.0
            and float(row["nonmyopic_gap_lcb95"]) > 0.0
            and int(row["min_meaningful_duration_branches"]) >= 2
            and float(row["max_dominant_duration_branch_share"]) <= 0.80
        )
        row["passes_gates"] = int(passed)
        row.pop("indices", None)
        rows.append(row)
        write_csv(diag / "validated_finalists.partial.csv", rows)
        print(
            f"validated {i}/{len(keys)} {row['durations_s']} "
            f"adaptive_lcb={row['adaptive_gap_lcb95']:.6f} "
            f"nonmyopic_lcb={row['nonmyopic_gap_lcb95']:.6f} pass={int(passed)}",
            flush=True,
        )
    rows.sort(
        key=lambda r: (
            int(r["passes_gates"]),
            min(float(r["adaptive_gap_lcb95"]), float(r["nonmyopic_gap_lcb95"])),
        ), reverse=True,
    )
    passing = [r for r in rows if int(r["passes_gates"])]
    write_csv(diag / "validated_finalists.csv", rows)
    write_csv(diag / "passing_candidates.csv", passing)
    summary = {
        **config, "status": "pass" if passing else "no_duration_set_passed",
        "n_passing": len(passing), "elapsed_seconds": time.time() - started,
        "selection_rule": "adaptive_lcb95>0; nonmyopic_lcb95>0; >=2 meaningful duration branches; max duration share<=0.80",
        "removed_redundant_gate": "lookahead duration differs from myopic for every individual seed",
        "obs_indices": obs_indices.tolist(),
    }
    (diag / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    config.update({"status": "complete", "n_passing": len(passing), "elapsed_seconds": summary["elapsed_seconds"]})
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"complete passing={len(passing)}/{len(rows)} results={diag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
