#!/usr/bin/env python3
"""Search the full 281-duration IEEE9 bank for six-duration EIG spaces.

The exact space C(281, 6) is too large to enumerate.  This tool uses every
0.01-s duration as an eligible coordinate, performs a reproducible global
search plus one-coordinate local refinement, and then applies the strict
three-seed adaptive/non-myopic duration-branching audit to the finalists.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
from tools.bank_sweeps.sweep_ieee9_eig_duration_sets import (
    Catalog,
    evaluate_combination_seed,
    load_catalog,
    load_support_centres,
    mean_lcb,
    precompute_seed,
    resolve_pool_actions,
    write_csv,
)


def candidate_key(indices: np.ndarray | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in indices))


def proxy_scores(centres: np.ndarray, n_durations: int, n_buses: int) -> tuple[np.ndarray, np.ndarray]:
    """Return duration informativeness and pairwise response dissimilarity."""
    shaped = centres.reshape(n_durations, n_buses, centres.shape[1], centres.shape[2])
    # Preserve particle-dependent response shape; remove the per-action mean so
    # duration separation is not merely a waveform offset.
    fingerprints = shaped - shaped.mean(axis=2, keepdims=True)
    fingerprints = fingerprints.reshape(n_durations, -1).astype(np.float64)
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    normalized = fingerprints / np.maximum(norms, 1e-12)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    dissimilarity = 1.0 - similarity
    info = shaped.var(axis=2).mean(axis=(1, 2)).astype(np.float64)
    info = info / max(float(np.max(info)), 1e-12)
    return info, dissimilarity


def proxy_value(key: tuple[int, ...], info: np.ndarray, distance: np.ndarray) -> float:
    ids = np.asarray(key, dtype=np.int64)
    pair = distance[np.ix_(ids, ids)]
    upper = pair[np.triu_indices(len(ids), 1)]
    # Require all six durations to be informative; reward non-redundant shapes.
    return float(0.55 * np.min(info[ids]) + 0.25 * np.mean(info[ids]) + 0.20 * np.mean(upper))


def exact_audit(
    key: tuple[int, ...],
    *,
    durations: tuple[float, ...],
    all_centres: np.ndarray,
    n_buses: int,
    sigma: float,
    seeds: tuple[int, ...],
    outer: int,
    inner: int,
    device: torch.device,
    min_branch_share: float,
    max_dominance: float,
) -> dict:
    action_ids = np.concatenate(
        [np.arange(i * n_buses, (i + 1) * n_buses, dtype=np.int64) for i in key]
    )
    centres = all_centres[action_ids]
    local_duration = np.repeat(np.arange(6, dtype=np.int64), n_buses)
    active = np.arange(6 * n_buses, dtype=np.int64)
    rows = []
    for seed in seeds:
        immediate, continuation = precompute_seed(
            centres,
            sigma=sigma,
            outer_fantasies=outer,
            inner_fantasies=inner,
            seed=seed,
            device=device,
        )
        rows.append(
            evaluate_combination_seed(
                active,
                immediate,
                continuation,
                pool_action_duration=local_duration,
                min_branch_share=min_branch_share,
            )
        )
    adaptive_mean, adaptive_lcb = mean_lcb([float(r["adaptive_gap"]) for r in rows])
    nonmyopic_mean, nonmyopic_lcb = mean_lcb([float(r["nonmyopic_gap"]) for r in rows])
    min_duration_branches = min(int(r["meaningful_duration_branches"]) for r in rows)
    max_duration_dominance = max(float(r["dominant_duration_branch_share"]) for r in rows)
    duration_differs_all = all(bool(r["first_duration_differs"]) for r in rows)
    passed = (
        adaptive_lcb > 0.0
        and nonmyopic_lcb > 0.0
        and min_duration_branches >= 2
        and max_duration_dominance <= max_dominance
        and duration_differs_all
    )
    return {
        "indices": key,
        "durations_s": ";".join(f"{durations[i]:.2f}" for i in key),
        "adaptive_gap_mean": adaptive_mean,
        "adaptive_gap_lcb95": adaptive_lcb,
        "nonmyopic_gap_mean": nonmyopic_mean,
        "nonmyopic_gap_lcb95": nonmyopic_lcb,
        "min_meaningful_duration_branches": min_duration_branches,
        "max_dominant_duration_branch_share": max_duration_dominance,
        "lookahead_duration_differs_all_seeds": int(duration_differs_all),
        "passes_gates": int(passed),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", default="data/ieee9_duration_dense_0p01")
    p.add_argument("--global-samples", type=int, default=20000)
    p.add_argument("--global-exact", type=int, default=120)
    p.add_argument("--local-parents", type=int, default=12)
    p.add_argument("--local-exact", type=int, default=120)
    p.add_argument("--finalists", type=int, default=20)
    p.add_argument("--search-seed", type=int, default=8675309)
    p.add_argument("--audit-seeds", default="101,202,303")
    p.add_argument("--N-obs", dest="n_obs", type=int, default=5)
    p.add_argument("--noise-sigma", type=float, default=0.01)
    p.add_argument("--support-size", type=int, default=96)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()

    bank_dir = (ROOT / args.bank_dir).resolve()
    catalog: Catalog = load_catalog(bank_dir)
    durations = tuple(round(0.20 + 0.01 * i, 2) for i in range(281))
    bank_pool_ids, _ = resolve_pool_actions(catalog, durations)
    n_buses = len(catalog.buses)
    centres, obs_indices = load_support_centres(
        bank_dir, bank_pool_ids, n_obs=args.n_obs,
        support_size=args.support_size, support_seed=7919,
    )
    if centres.shape[0] != 281 * n_buses:
        raise RuntimeError(f"expected {281*n_buses} actions, got {centres.shape[0]}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    seeds = tuple(int(x) for x in args.audit_seeds.split(",") if x.strip())
    folder = make_experiment_dir_name(
        "ieee9_duration_fullgrid_search", "eig_based", 2,
        n_obs=args.n_obs, noise_sigma=args.noise_sigma,
    )
    out = ROOT / "experiments" / folder
    diag = out / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run = {
        "system": "ieee9", "study_type": "full_281_duration_global_local_search",
        "duration_grid_s": [0.20, 3.00, 0.01], "n_duration_options": 281,
        "combination_space": math.comb(281, 6), "choose": 6,
        "global_samples": args.global_samples, "global_exact": args.global_exact,
        "local_parents": args.local_parents, "local_exact": args.local_exact,
        "finalists": args.finalists, "audit_seeds": list(seeds),
        "N_obs": args.n_obs, "noise_sigma": args.noise_sigma,
        "status": "running",
    }
    (out / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"full grid=281, C(281,6)={math.comb(281,6):,}, device={device}", flush=True)

    info, distance = proxy_scores(centres, 281, n_buses)
    rng = np.random.default_rng(args.search_seed)
    candidates: set[tuple[int, ...]] = set()
    # Random coverage plus stratified coverage ensures the whole interval and
    # every duration coordinate can enter the search.
    for _ in range(args.global_samples):
        candidates.add(candidate_key(rng.choice(281, 6, replace=False)))
    for offset in range(281):
        companions = rng.choice(np.delete(np.arange(281), offset), 5, replace=False)
        candidates.add(candidate_key(np.r_[offset, companions]))
    proxy_ranked = sorted(
        candidates, key=lambda k: proxy_value(k, info, distance), reverse=True
    )
    global_keys = proxy_ranked[: args.global_exact]
    print(f"global proxy={len(candidates)}, exact={len(global_keys)}", flush=True)
    low_rows = []
    for i, key in enumerate(global_keys, 1):
        row = exact_audit(
            key, durations=durations, all_centres=centres, n_buses=n_buses,
            sigma=args.noise_sigma, seeds=(seeds[0],), outer=8, inner=4,
            device=device, min_branch_share=0.10, max_dominance=0.80,
        )
        row["proxy_score"] = proxy_value(key, info, distance)
        low_rows.append(row)
        if i % 20 == 0 or i == len(global_keys):
            print(f"global exact {i}/{len(global_keys)}", flush=True)
    low_rows.sort(
        key=lambda r: min(r["adaptive_gap_mean"], r["nonmyopic_gap_mean"]),
        reverse=True,
    )

    local: set[tuple[int, ...]] = set()
    for parent in low_rows[: args.local_parents]:
        base = tuple(parent["indices"])
        for position in range(6):
            for replacement in range(281):
                if replacement not in base:
                    trial = list(base); trial[position] = replacement
                    local.add(candidate_key(trial))
    local_ranked = sorted(
        local, key=lambda k: proxy_value(k, info, distance), reverse=True
    )[: args.local_exact]
    print(f"local neighbors={len(local)}, exact={len(local_ranked)}", flush=True)
    for i, key in enumerate(local_ranked, 1):
        row = exact_audit(
            key, durations=durations, all_centres=centres, n_buses=n_buses,
            sigma=args.noise_sigma, seeds=(seeds[0],), outer=8, inner=4,
            device=device, min_branch_share=0.10, max_dominance=0.80,
        )
        row["proxy_score"] = proxy_value(key, info, distance)
        low_rows.append(row)
        if i % 20 == 0 or i == len(local_ranked):
            print(f"local exact {i}/{len(local_ranked)}", flush=True)

    best_by_key = {}
    for row in low_rows:
        key = tuple(row["indices"])
        score = min(float(row["adaptive_gap_mean"]), float(row["nonmyopic_gap_mean"]))
        if key not in best_by_key or score > best_by_key[key][0]:
            best_by_key[key] = (score, row)
    finalists = [v[1] for v in sorted(best_by_key.values(), reverse=True, key=lambda x: x[0])[: args.finalists]]
    final_rows = []
    for i, prior in enumerate(finalists, 1):
        key = tuple(prior["indices"])
        row = exact_audit(
            key, durations=durations, all_centres=centres, n_buses=n_buses,
            sigma=args.noise_sigma, seeds=seeds, outer=16, inner=8,
            device=device, min_branch_share=0.10, max_dominance=0.80,
        )
        row["proxy_score"] = proxy_value(key, info, distance)
        row.pop("indices", None)
        final_rows.append(row)
        print(f"strict finalist {i}/{len(finalists)}: {row['durations_s']} pass={row['passes_gates']}", flush=True)
    for row in low_rows:
        row.pop("indices", None)
    final_rows.sort(
        key=lambda r: (int(r["passes_gates"]), min(float(r["adaptive_gap_lcb95"]), float(r["nonmyopic_gap_lcb95"]))),
        reverse=True,
    )
    passing = [r for r in final_rows if int(r["passes_gates"])]
    write_csv(diag / "search_candidates.csv", low_rows)
    write_csv(diag / "strict_finalists.csv", final_rows)
    write_csv(diag / "passing_candidates.csv", passing)
    summary = {
        **run, "status": "pass" if passing else "no_duration_set_passed",
        "n_global_candidates": len(candidates), "n_local_neighbors": len(local),
        "n_strict_finalists": len(final_rows), "n_passing": len(passing),
        "selection_rule": "adaptive_lcb95>0; nonmyopic_lcb95>0; >=2 duration branches; max duration share<=0.80; lookahead duration differs from myopic for every seed",
        "search_is_exhaustive": False,
        "search_note": "All 281 durations eligible; reproducible global random coverage plus one-coordinate local refinement.",
        "obs_indices": obs_indices.tolist(), "elapsed_seconds": time.time() - started,
    }
    (diag / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run.update({"status": "complete", "n_passing": len(passing), "elapsed_seconds": summary["elapsed_seconds"]})
    (out / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"complete passing={len(passing)}/{len(final_rows)} results={diag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
