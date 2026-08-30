#!/usr/bin/env python3
"""Screen six-duration IEEE9 EIG design spaces from one dense physical bank.

This is a standalone, method-independent diagnostic.  It does not train DAD,
RL-sBOED, or any other policy, and it does not modify the production bank.

For a duration set D, the active action space is every (duration, bus) pair in
D x probe_buses.  The tool estimates, for a two-experiment horizon:

* adaptive gap: best adaptive value minus best open-loop pair value;
* non-myopic gap: best lookahead-first value minus the value obtained when the
  first action is forced to be the one-step-myopic action;
* duration branching: whether different observations change the optimal next
  duration (changing only the bus does not count as adaptive duration space).

Expensive action/fantasy values are precomputed once per seed.  All unordered
duration combinations are then reductions of those cached arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.objectives.eig.vector import VectorEIGEngine
from src.observations.likelihood import evenly_spaced_indices
from src.layout import make_experiment_dir_name


DEFAULT_DURATIONS = tuple(np.round(np.arange(0.2, 3.0001, 0.2), 2))


@dataclass(frozen=True)
class Catalog:
    designs: tuple[tuple[float, int, float], ...]

    @property
    def buses(self) -> tuple[int, ...]:
        return tuple(sorted({int(row[1]) for row in self.designs}))

    @property
    def durations(self) -> tuple[float, ...]:
        return tuple(sorted({round(float(row[2]), 10) for row in self.designs}))


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def load_catalog(bank_dir: Path) -> Catalog:
    path = bank_dir / "meta" / "catalog.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    designs = tuple(
        (float(row[0]), int(row[1]), float(row[2])) for row in raw["designs"]
    )
    if not designs:
        raise ValueError(f"empty action catalog: {path}")
    return Catalog(designs=designs)


def resolve_pool_actions(
    catalog: Catalog,
    durations: Iterable[float],
    *,
    tolerance: float = 5e-7,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    requested = tuple(sorted(set(round(float(x), 10) for x in durations)))
    by_duration: dict[float, np.ndarray] = {}
    for duration in requested:
        ids = np.asarray(
            [
                i
                for i, (_, _, bank_duration) in enumerate(catalog.designs)
                if abs(float(bank_duration) - duration) <= tolerance
            ],
            dtype=np.int64,
        )
        if ids.size != len(catalog.buses):
            raise ValueError(
                f"duration {duration:g}s maps to {ids.size} bank actions; "
                f"expected one action for each of {len(catalog.buses)} buses"
            )
        by_duration[duration] = ids
    pool_ids = np.concatenate([by_duration[d] for d in requested])
    return pool_ids, by_duration


def load_support_centres(
    bank_dir: Path,
    bank_action_ids: np.ndarray,
    *,
    n_obs: int,
    support_size: int,
    support_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = bank_dir / "train" / "delta_f.npy"
    full = np.load(path, mmap_mode="r")
    if full.ndim != 3:
        raise ValueError(f"expected theta x action x time bank, got {full.shape}")
    if int(bank_action_ids.max()) >= full.shape[1]:
        raise ValueError("catalog action ID exceeds the delta_f bank action dimension")
    obs_indices = evenly_spaced_indices(int(full.shape[2]), int(n_obs))
    n_support = min(int(support_size), int(full.shape[0]))
    rng = np.random.default_rng(int(support_seed))
    particle_ids = np.sort(rng.choice(full.shape[0], n_support, replace=False))
    # np.ix_ makes one compact copy: support x pool_action x observation.
    selected = np.asarray(
        full[np.ix_(particle_ids, bank_action_ids, obs_indices)], dtype=np.float32
    )
    centres = np.transpose(selected, (1, 0, 2))
    return centres, obs_indices


@torch.no_grad()
def precompute_seed(
    centres_np: np.ndarray,
    *,
    sigma: float,
    outer_fantasies: int,
    inner_fantasies: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return immediate[A] and conditional continuation[A,F,A]."""
    n_actions, n_particles, _ = centres_np.shape
    context = type("SweepEIGContext", (), {})()
    context.centres_support = centres_np
    context.log_p0 = np.full(n_particles, -math.log(n_particles), dtype=np.float64)
    context.sigma_y = float(sigma)
    context.n_actions = int(n_actions)
    engine = VectorEIGEngine(context, device)
    log_p0 = engine.log_p0.clone()
    feasible_all = np.arange(n_actions, dtype=np.int64)
    immediate = engine.action_scores(
        log_p0,
        feasible_all,
        n_fantasies=max(int(outer_fantasies) * 4, 64),
        seed=int(seed),
    )
    continuation = torch.empty(
        (n_actions, int(outer_fantasies), n_actions),
        dtype=torch.float32,
        device="cpu",
    )
    prior_p = torch.softmax(log_p0, dim=-1)
    for first in range(n_actions):
        first_generator = torch.Generator(device=device).manual_seed(
            int(seed) + 104_729 * (first + 1)
        )
        particle_ids = torch.multinomial(
            prior_p,
            int(outer_fantasies),
            replacement=True,
            generator=first_generator,
        )
        observations = engine.centres[particle_ids, first, :] + torch.randn(
            (int(outer_fantasies), engine.centres.shape[2]),
            dtype=engine.centres.dtype,
            device=device,
            generator=first_generator,
        ) * float(sigma)
        for fantasy in range(int(outer_fantasies)):
            posterior = engine.update(log_p0, first, observations[fantasy])
            feasible_next = feasible_all[feasible_all != first]
            gains = engine.action_scores(
                posterior,
                feasible_next,
                n_fantasies=int(inner_fantasies),
                seed=int(seed) + 1_000_003 * (first + 1) + fantasy,
            )
            continuation[first, fantasy] = torch.as_tensor(gains)
        if (first + 1) % 10 == 0 or first + 1 == n_actions:
            print(
                f"  seed={seed}: first-action values {first + 1}/{n_actions}",
                flush=True,
            )
    return np.asarray(immediate, dtype=np.float64), continuation.numpy()


def active_pool_ids(
    combination: tuple[float, ...],
    duration_to_pool_ids: dict[float, np.ndarray],
) -> np.ndarray:
    return np.concatenate([duration_to_pool_ids[round(float(d), 10)] for d in combination])


def evaluate_combination_seed(
    action_ids: np.ndarray,
    immediate: np.ndarray,
    continuation: np.ndarray,
    *,
    pool_action_duration: np.ndarray,
    min_branch_share: float,
) -> dict[str, float | int]:
    ids = np.asarray(action_ids, dtype=np.int64)
    imm = immediate[ids]
    myopic_local = int(np.argmax(imm))
    myopic_first = int(ids[myopic_local])
    myopic_first_duration = int(pool_action_duration[myopic_first])
    adaptive_q = np.empty(ids.size, dtype=np.float64)
    fixed_q = np.empty((ids.size, ids.size), dtype=np.float64)
    for left, first in enumerate(ids):
        conditional = continuation[first][:, ids].astype(np.float64, copy=True)
        conditional[:, left] = -np.inf
        adaptive_q[left] = float(immediate[first]) + float(
            np.max(conditional, axis=1).mean()
        )
        fixed_q[left] = float(immediate[first]) + conditional.mean(axis=0)
        fixed_q[left, left] = -np.inf
    lookahead_local = int(np.argmax(adaptive_q))
    lookahead_first = int(ids[lookahead_local])
    lookahead_first_duration = int(pool_action_duration[lookahead_first])
    adaptive_value = float(adaptive_q[lookahead_local])
    fixed_value = float(np.max(fixed_q))
    myopic_value = float(adaptive_q[myopic_local])

    branch_values = continuation[lookahead_first][:, ids].astype(np.float64, copy=True)
    branch_values[:, lookahead_local] = -np.inf
    branch_local = np.argmax(branch_values, axis=1)
    counts = np.bincount(branch_local, minlength=ids.size)
    shares = counts / max(int(counts.sum()), 1)
    positive = shares[shares > 0]
    branch_entropy = float(-np.sum(positive * np.log(positive)))
    meaningful = int(np.count_nonzero(shares >= float(min_branch_share)))
    branch_pool_ids = ids[branch_local]
    branch_durations = pool_action_duration[branch_pool_ids]
    duration_counts = np.bincount(
        branch_durations, minlength=int(pool_action_duration.max()) + 1
    )
    duration_shares = duration_counts / max(int(duration_counts.sum()), 1)
    positive_duration = duration_shares[duration_shares > 0]
    duration_entropy = float(
        -np.sum(positive_duration * np.log(positive_duration))
    )
    meaningful_durations = int(
        np.count_nonzero(duration_shares >= float(min_branch_share))
    )
    return {
        "adaptive_value": adaptive_value,
        "fixed_value": fixed_value,
        "myopic_value": myopic_value,
        "adaptive_gap": adaptive_value - fixed_value,
        "nonmyopic_gap": adaptive_value - myopic_value,
        "myopic_first_pool_id": myopic_first,
        "lookahead_first_pool_id": lookahead_first,
        "first_action_differs": int(myopic_first != lookahead_first),
        "first_duration_differs": int(
            myopic_first_duration != lookahead_first_duration
        ),
        "branch_entropy": branch_entropy,
        "meaningful_branches": meaningful,
        "dominant_branch_share": float(shares.max()),
        "duration_branch_entropy": duration_entropy,
        "meaningful_duration_branches": meaningful_durations,
        "dominant_duration_branch_share": float(duration_shares.max()),
    }


def mean_lcb(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean
    sem = float(array.std(ddof=1) / math.sqrt(array.size))
    return mean, mean - 1.96 * sem


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    """Prefer a project-relative path, but allow /tmp smoke-test outputs."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank-dir", default="data/ieee9_duration_dense_0p01"
    )
    parser.add_argument(
        "--durations",
        type=parse_float_list,
        default=DEFAULT_DURATIONS,
        help="Candidate duration pool; default is 0.2,0.4,...,3.0 seconds.",
    )
    parser.add_argument("--choose", type=int, default=6)
    parser.add_argument("--N-obs", "--n-obs", dest="n_obs", type=int, default=5)
    parser.add_argument("--noise-sigma", type=float, default=0.01)
    parser.add_argument("--support-size", type=int, default=96)
    parser.add_argument("--outer-fantasies", type=int, default=16)
    parser.add_argument("--inner-fantasies", type=int, default=8)
    parser.add_argument("--seeds", type=parse_int_list, default=(101, 202, 303))
    parser.add_argument("--min-branch-share", type=float, default=0.10)
    parser.add_argument("--min-meaningful-branches", type=int, default=2)
    parser.add_argument("--min-meaningful-duration-branches", type=int, default=2)
    parser.add_argument("--max-dominant-duration-share", type=float, default=0.80)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Explicit output directory, mainly for /tmp smoke tests. By default "
            "a unified stamped experiments/..._EIG_T2_Nobs#_sigma# folder is created."
        ),
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=0,
        help="Optional deterministic prefix limit for smoke tests; 0 means all.",
    )
    args = parser.parse_args()

    if args.choose < 2:
        raise SystemExit("--choose must be at least 2")
    durations = tuple(sorted(set(round(float(x), 10) for x in args.durations)))
    if len(durations) < args.choose:
        raise SystemExit("candidate duration pool is smaller than --choose")
    if args.noise_sigma <= 0:
        raise SystemExit("--noise-sigma must be positive")
    bank_dir = (ROOT / args.bank_dir).resolve()
    if args.output_dir:
        experiment_dir = (ROOT / args.output_dir).resolve()
        output_dir = experiment_dir
        cache_dir = output_dir / "cache"
    else:
        folder_name = make_experiment_dir_name(
            "ieee9_duration_combo_sweep",
            "eig_based",
            2,
            n_obs=int(args.n_obs),
            noise_sigma=float(args.noise_sigma),
        )
        experiment_dir = (ROOT / "experiments" / folder_name).resolve()
        output_dir = experiment_dir / "diagnostics"
        cache_dir = experiment_dir / "scratch" / "duration_sweep_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(bank_dir)
    bank_pool_ids, duration_to_bank_ids = resolve_pool_actions(catalog, durations)
    bank_to_pool = {int(bank_id): i for i, bank_id in enumerate(bank_pool_ids)}
    duration_to_pool_ids = {
        duration: np.asarray([bank_to_pool[int(x)] for x in bank_ids], dtype=np.int64)
        for duration, bank_ids in duration_to_bank_ids.items()
    }
    pool_action_duration = np.empty(len(bank_pool_ids), dtype=np.int64)
    for duration_index, duration in enumerate(durations):
        pool_action_duration[duration_to_pool_ids[duration]] = duration_index
    combinations = list(itertools.combinations(durations, int(args.choose)))
    if args.max_combinations > 0:
        combinations = combinations[: int(args.max_combinations)]
    expected_actions = int(args.choose) * len(catalog.buses)
    print(
        f"bank={bank_dir}\n"
        f"duration_pool={len(durations)}, combinations={len(combinations)}, "
        f"active_actions_per_combination={expected_actions}",
        flush=True,
    )
    run_config_path = experiment_dir / "run_config.json"
    run_config = {
        "system": "ieee9",
        "system_label": "IEEE-9",
        "experiment_type": "eig_based",
        "study_type": "duration_combo_structural_sweep",
        "T": 2,
        "step_number": 2,
        "N_obs": int(args.n_obs),
        "noise_sigma": float(args.noise_sigma),
        "methods": [],
        "bank_dir": str(bank_dir.relative_to(ROOT)),
        "duration_pool_s": list(durations),
        "choose": int(args.choose),
        "n_buses": len(catalog.buses),
        "n_active_actions": expected_actions,
        "n_combinations_requested": len(combinations),
        "support_size": int(args.support_size),
        "outer_fantasies": int(args.outer_fantasies),
        "inner_fantasies": int(args.inner_fantasies),
        "seeds": list(args.seeds),
        "status": "running",
    }
    run_config_path.write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    centres, obs_indices = load_support_centres(
        bank_dir,
        bank_pool_ids,
        n_obs=args.n_obs,
        support_size=args.support_size,
        support_seed=7_919,
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"device={device}, centres={centres.shape}, obs_indices={obs_indices.tolist()}")
    seed_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pool_tag = hashlib.sha256(
        bank_pool_ids.tobytes() + obs_indices.tobytes()
    ).hexdigest()[:12]
    for seed in args.seeds:
        cache = cache_dir / (
            f"values_{pool_tag}_seed{seed}_P{centres.shape[1]}_O{args.outer_fantasies}_"
            f"I{args.inner_fantasies}_Nobs{args.n_obs}_sigma{args.noise_sigma:g}.npz"
        )
        if cache.is_file():
            payload = np.load(cache)
            immediate = payload["immediate"]
            continuation = payload["continuation"]
            print(f"loaded {display_path(cache)}", flush=True)
        else:
            started = time.perf_counter()
            immediate, continuation = precompute_seed(
                centres,
                sigma=args.noise_sigma,
                outer_fantasies=args.outer_fantasies,
                inner_fantasies=args.inner_fantasies,
                seed=seed,
                device=device,
            )
            np.savez_compressed(cache, immediate=immediate, continuation=continuation)
            print(
                f"wrote {display_path(cache)} in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if immediate.shape != (len(bank_pool_ids),) or continuation.shape != (
            len(bank_pool_ids),
            args.outer_fantasies,
            len(bank_pool_ids),
        ):
            raise RuntimeError(f"stale/incompatible cache shape in {cache}")
        seed_values[int(seed)] = (immediate, continuation)

    rows: list[dict] = []
    for index, combination in enumerate(combinations):
        active = active_pool_ids(combination, duration_to_pool_ids)
        if active.size != expected_actions:
            raise RuntimeError(f"{combination} produced {active.size} active actions")
        per_seed = [
            evaluate_combination_seed(
                active,
                *seed_values[int(seed)],
                pool_action_duration=pool_action_duration,
                min_branch_share=args.min_branch_share,
            )
            for seed in args.seeds
        ]
        adaptive_mean, adaptive_lcb = mean_lcb(
            [float(row["adaptive_gap"]) for row in per_seed]
        )
        nonmyopic_mean, nonmyopic_lcb = mean_lcb(
            [float(row["nonmyopic_gap"]) for row in per_seed]
        )
        min_branches = min(int(row["meaningful_branches"]) for row in per_seed)
        min_duration_branches = min(
            int(row["meaningful_duration_branches"]) for row in per_seed
        )
        max_duration_dominance = max(
            float(row["dominant_duration_branch_share"]) for row in per_seed
        )
        differs_all = all(bool(row["first_action_differs"]) for row in per_seed)
        duration_differs_all = all(
            bool(row["first_duration_differs"]) for row in per_seed
        )
        passed = bool(
            adaptive_lcb > 0.0
            and nonmyopic_lcb > 0.0
            and min_branches >= int(args.min_meaningful_branches)
            and differs_all
            and min_duration_branches
            >= int(args.min_meaningful_duration_branches)
            and max_duration_dominance
            <= float(args.max_dominant_duration_share)
            and duration_differs_all
        )
        bank_action_ids = bank_pool_ids[active]
        rows.append(
            {
                "durations_s": ";".join(f"{x:.2f}" for x in combination),
                "n_active_actions": int(active.size),
                "adaptive_gap_mean": adaptive_mean,
                "adaptive_gap_lcb95": adaptive_lcb,
                "nonmyopic_gap_mean": nonmyopic_mean,
                "nonmyopic_gap_lcb95": nonmyopic_lcb,
                "min_meaningful_branches": min_branches,
                "mean_branch_entropy": float(
                    np.mean([float(row["branch_entropy"]) for row in per_seed])
                ),
                "max_dominant_branch_share": max(
                    float(row["dominant_branch_share"]) for row in per_seed
                ),
                "min_meaningful_duration_branches": min_duration_branches,
                "mean_duration_branch_entropy": float(
                    np.mean(
                        [float(row["duration_branch_entropy"]) for row in per_seed]
                    )
                ),
                "max_dominant_duration_branch_share": max_duration_dominance,
                "lookahead_differs_from_myopic_all_seeds": int(differs_all),
                "lookahead_duration_differs_from_myopic_all_seeds": int(
                    duration_differs_all
                ),
                "passes_gates": int(passed),
                "bank_action_ids": ";".join(str(int(x)) for x in bank_action_ids),
            }
        )
        if (index + 1) % 500 == 0 or index + 1 == len(combinations):
            print(f"scored combinations {index + 1}/{len(combinations)}", flush=True)

    # Conservative ordering: pass first, then weakest of the two lower bounds.
    rows.sort(
        key=lambda row: (
            int(row["passes_gates"]),
            min(
                float(row["adaptive_gap_lcb95"]),
                float(row["nonmyopic_gap_lcb95"]),
            ),
            float(row["adaptive_gap_lcb95"]) + float(row["nonmyopic_gap_lcb95"]),
        ),
        reverse=True,
    )
    passing = [row for row in rows if int(row["passes_gates"]) == 1]
    top = passing[: int(args.top)]
    write_csv(output_dir / "all_candidates.csv", rows)
    write_csv(output_dir / "passing_candidates.csv", passing)
    write_csv(output_dir / f"top{args.top}.csv", top)
    summary = {
        "purpose": "method-independent IEEE9 six-duration EIG space sweep",
        "bank_dir": str(bank_dir.relative_to(ROOT)),
        "duration_pool_s": list(durations),
        "choose": int(args.choose),
        "n_buses": len(catalog.buses),
        "n_active_actions": expected_actions,
        "n_combinations_scored": len(rows),
        "n_passing": len(passing),
        "n_selected": len(top),
        "selection_rule": (
            "adaptive_gap_lcb95>0 AND nonmyopic_gap_lcb95>0 AND "
            f"meaningful_branches>={args.min_meaningful_branches} AND "
            f"meaningful_duration_branches>={args.min_meaningful_duration_branches} "
            f"AND dominant_duration_share<={args.max_dominant_duration_share:g} "
            "AND lookahead_duration!=myopic_duration for every seed"
        ),
        "seeds": list(args.seeds),
        "support_size": int(centres.shape[1]),
        "outer_fantasies": int(args.outer_fantasies),
        "inner_fantasies": int(args.inner_fantasies),
        "N_obs": int(args.n_obs),
        "noise_sigma": float(args.noise_sigma),
        "obs_indices": obs_indices.tolist(),
        "device": str(device),
        "status": "pass" if passing else "no_duration_set_passed",
        "no_pass_action": (
            "Do not choose a least-bad set; run a full-grid candidate search or "
            "redesign observations/actions."
            if not passing
            else "Validate selected candidates with larger Monte Carlo budgets."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_config.update(
        {
            "status": "complete",
            "n_combinations_scored": len(rows),
            "n_passing": len(passing),
            "n_selected": len(top),
            "diagnostics_dir": str(output_dir.relative_to(experiment_dir)),
        }
    )
    run_config_path.write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"completed: passing={len(passing)}/{len(rows)}, selected={len(top)}; "
        f"results={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
