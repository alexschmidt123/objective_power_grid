"""Generate final publication tables from crossed training/evaluation runs.

Trained methods use every 3-training-seed x 5-evaluation-seed result. Baselines
are deduplicated by evaluation seed. Offline time is summarized at run level.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.layout import parse_result_dir_name

TRAIN_SEEDS = (101, 202, 303)
EVAL_SEEDS = (1001, 1002, 1003, 1004, 1005)
TRAINED = {"DAD", "RL-SBOED", "Step-DAD"}
METHOD_ORDER = ("DAD", "RL-SBOED", "Step-DAD", "Myopic", "Fixed", "Random")
LABELS = {
    "dad": "DAD", "dad_eig": "DAD", "DAD": "DAD",
    "rl_sboed": "RL-SBOED", "rl_sboed_eig": "RL-SBOED",
    "RL-sBOED": "RL-SBOED", "RL-SBOED": "RL-SBOED",
    "step_dad": "Step-DAD", "Step-DAD": "Step-DAD",
    "myopic": "Myopic", "myopic_delta_h": "Myopic", "Myopic": "Myopic",
    "fixed": "Fixed", "fixed_open_loop": "Fixed", "Fixed": "Fixed",
    "random": "Random", "Random": "Random",
}

Crossed = dict[tuple[int, int], list[float]]


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return _mean(values), statistics.stdev(values)


def _float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw not in (None, ""):
            return float(raw)
    return 0.0


def _training_seed(exp_dir: Path) -> int:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", exp_dir.name)
    if not match:
        raise ValueError(f"training seed missing from result folder: {exp_dir}")
    return int(match.group(1))


def _evaluation_seed(path: Path, row: dict[str, Any]) -> int:
    raw = row.get("eval_seed")
    if raw not in (None, ""):
        return int(raw)
    match = re.search(r"seed_(\d+)", path.as_posix())
    if not match:
        raise ValueError(f"evaluation seed missing: {path}")
    return int(match.group(1))


def collect(
    exp_dirs: list[Path], *, eig: bool
) -> tuple[dict[tuple[str, int], Crossed], dict[tuple[str, int], dict[int, float]], dict[int, set[int]]]:
    crossed: dict[tuple[str, int], Crossed] = defaultdict(lambda: defaultdict(list))
    offline: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    coverage: dict[int, set[int]] = defaultdict(set)
    filename = "terminal_eig_summary.csv" if eig else "summary.csv"
    metric_keys = ("terminal_eig_mean", "mean_eig", "ΔH") if eig else ("mean_mocu", "mean_gap")
    for exp_dir in exp_dirs:
        parsed = parse_result_dir_name(exp_dir.name)
        if parsed is None:
            raise ValueError(f"invalid result folder name: {exp_dir.name}")
        horizon = int(parsed["T"])
        train_seed = _training_seed(exp_dir)
        coverage[horizon].add(train_seed)
        paths = sorted((exp_dir / "evaluations").glob(f"seed_*/eval/{filename}"))
        for path in paths:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    method = LABELS.get(str(row.get("method") or row.get("Method")))
                    if method not in METHOD_ORDER:
                        continue
                    eval_seed = _evaluation_seed(path, row)
                    crossed[(method, horizon)][(train_seed, eval_seed)].append(
                        _float(row, *metric_keys)
                    )
                    crossed[(method + "::online", horizon)][(train_seed, eval_seed)].append(
                        _float(row, "online_seconds_per_rollout", "seconds_per_rollout")
                    )
                    offline[(method, horizon)][train_seed] = _float(
                        row, "training_time_seconds",
                        "offline_training_or_calibration_seconds", "train_s"
                    )
    return crossed, offline, coverage


def crossed_values(values: Crossed, method: str) -> list[float] | None:
    required = {(tr, ev) for tr in TRAIN_SEEDS for ev in EVAL_SEEDS}
    if set(values) != required:
        return None
    if method in TRAINED:
        return [_mean(values[(tr, ev)]) for tr in TRAIN_SEEDS for ev in EVAL_SEEDS]
    return [
        _mean([_mean(values[(tr, ev)]) for tr in TRAIN_SEEDS])
        for ev in EVAL_SEEDS
    ]


def _cell(values: list[float] | None, digits: int) -> tuple[str, float | None]:
    if values is None or len(values) < 2:
        return "—", None
    mean, sd = _mean_sd(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}", mean


def write_table(
    path: Path, title: str, horizons: list[int], cells: dict[tuple[str, int], list[float] | None],
    *, digits: int, higher_is_better: bool,
) -> None:
    rendered: dict[tuple[str, int], tuple[str, float | None]] = {
        key: _cell(value, digits) for key, value in cells.items()
    }
    winners: dict[int, set[str]] = {}
    for horizon in horizons:
        available = {m: rendered.get((m, horizon), ("—", None))[1] for m in METHOD_ORDER}
        available = {m: v for m, v in available.items() if v is not None}
        if available:
            target = (max if higher_is_better else min)(available.values())
            winners[horizon] = {m for m, v in available.items() if math.isclose(v, target)}
    lines = [f"# {title}", "", "| Method | " + " | ".join(f"T={t}" for t in horizons) + " |",
             "| --- | " + " | ".join("---:" for _ in horizons) + " |"]
    for method in METHOD_ORDER:
        row=[]
        for horizon in horizons:
            text = rendered.get((method, horizon), ("—", None))[0]
            if method in winners.get(horizon, set()):
                text = f"**{text}**"
            row.append(text)
        lines.append("| " + method + " | " + " | ".join(row) + " |")
    lines.extend(["", "Trained methods: 3 training seeds × 5 evaluation seeds. "
                  "Baselines: 5 unique evaluation seeds. Offline time: 3 run-level values.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(exp_dirs: list[Path], output_dir: Path, *, eig: bool) -> None:
    crossed, offline, coverage = collect(exp_dirs, eig=eig)
    horizons = sorted(coverage)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric: dict[tuple[str, int], list[float] | None] = {}
    online: dict[tuple[str, int], list[float] | None] = {}
    offline_cells: dict[tuple[str, int], list[float] | None] = {}
    for method in METHOD_ORDER:
        for horizon in horizons:
            metric[(method, horizon)] = crossed_values(crossed.get((method, horizon), {}), method)
            online[(method, horizon)] = crossed_values(crossed.get((method + "::online", horizon), {}), method)
            run_values = offline.get((method, horizon), {})
            offline_cells[(method, horizon)] = (
                [run_values[s] for s in TRAIN_SEEDS] if set(run_values) == set(TRAIN_SEEDS) else None
            )
    metric_name = "EIG" if eig else "MOCU"
    write_table(output_dir / f"{metric_name.lower()}_table.md", metric_name, horizons, metric,
                digits=4, higher_is_better=eig)
    write_table(output_dir / "offline_time_table.md", "Offline time (seconds)", horizons,
                offline_cells, digits=2, higher_is_better=False)
    write_table(output_dir / "online_time_table.md", "Online time (seconds per rollout)", horizons,
                online, digits=6, higher_is_better=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-type", choices=("eig_based", "objective_based"), required=True)
    parser.add_argument("--exp-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args=parser.parse_args()
    generate(args.exp_dir, args.output_dir, eig=args.experiment_type == "eig_based")


if __name__ == "__main__":
    main()
