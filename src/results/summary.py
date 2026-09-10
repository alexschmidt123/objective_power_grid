"""Write a single experiment-root ``summary.md`` comparison table.

Observation mode follows ``N_obs`` (max_rocof if 0, sampled Δf otherwise).
Primary metric: ``mean_posterior_mocu`` for objective_based, ``mean_eig`` for eig_based.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from src.observations.compress import observation_mode


def _fmt(value: Any, *, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |",
    ]
    for row in rows:
        cells = list(row) + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return lines


def _observation_blurb(n_obs: int, mode: str) -> str:
    if int(n_obs) == 0 or mode == "max_rocof":
        return (
            "Observation: scalar max-|ROCOF| (`N_obs=0`). "
            "Methods do not see full Δf trajectories."
        )
    return (
        f"Observation: {int(n_obs)} evenly spaced probe-bus Δf samples "
        f"(`observation_mode={mode}`)."
    )


def write_summary_md(
    exp_dir: Path,
    *,
    system: str,
    experiment_type: str,
    meta: dict[str, Any] | None = None,
    table_headers: Sequence[str],
    table_rows: Sequence[Sequence[str]],
    extra_lines: Sequence[str] | None = None,
) -> Path:
    """Write ``{exp_dir}/summary.md`` (no ``summary/`` folder)."""
    exp_dir = Path(exp_dir)
    meta = dict(meta or {})
    n_obs = int(meta.get("N_obs", meta.get("n_obs", 0)) or 0)
    mode = str(meta.get("observation_mode") or observation_mode(n_obs))
    n_sim = meta.get("N_sim", meta.get("n_sim"))
    step_number = meta.get("T", meta.get("step_number"))

    lines = [
        f"# Summary — {system} ({experiment_type})",
        "",
        _observation_blurb(n_obs, mode),
        "",
        f"- system: `{system}`",
        f"- experiment_type: `{experiment_type}`",
        f"- observation_mode: `{mode}`",
        f"- N_obs: {n_obs}",
    ]
    if n_sim is not None:
        lines.append(f"- N_sim: {n_sim}")
    if step_number is not None:
        lines.append(f"- T: {step_number}")
    if meta.get("config_path"):
        lines.append(f"- config: `{meta['config_path']}`")
    lines += ["", "## Comparison", ""]
    lines.extend(_md_table(table_headers, table_rows))
    if extra_lines:
        lines += ["", *extra_lines]
    lines.append("")

    out = exp_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_objective_summary_md(
    exp_dir: Path,
    *,
    system: str | None = None,
    eval_meta: dict[str, Any] | None = None,
) -> Path:
    """Build objective_based table from ``eval/summary.csv`` (+ Oracle)."""
    exp_dir = Path(exp_dir)
    eval_dir = exp_dir / "eval"
    summary_csv = eval_dir / "summary.csv"
    meta: dict[str, Any] = {}
    meta_path = eval_dir / "eval_meta.json"
    if meta_path.is_file():
        meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
    if eval_meta:
        meta.update(eval_meta)

    system = system or str(meta.get("system") or exp_dir.name)
    if "T" not in meta and "step_number" not in meta:
        from src.layout import parse_result_dir_name

        parsed = parse_result_dir_name(exp_dir.name)
        if parsed and parsed.get("step_number") is not None:
            meta["T"] = parsed["step_number"]

    parsed_rows = []
    if summary_csv.is_file():
        with summary_csv.open(encoding="utf-8") as handle:
            parsed_rows = [r for r in csv.DictReader(handle)
                           if r.get("method") != "Oracle"
                           and not r.get("method", "").endswith("_stochastic")]
    rows = [[str(r["method"]), _fmt(r.get("mean_posterior_mocu"))]
            for r in parsed_rows]
    extra = ["", "## Empirical safety rate", "",
             "| Method | Safety rate | Safe outcomes / evaluated outcomes | Physical systems |",
             "|---|---:|---:|---:|"]
    for r in parsed_rows:
        rate = r.get("safety_rate")
        rate_text = f"{100 * float(rate):.2f}%" if rate not in (None, "") else "—"
        safe = r.get("safety_safe_outcomes", "—")
        total = r.get("safety_n_outcomes", r.get("n_design_replicates", "—"))
        systems = r.get("safety_n_systems", r.get("n", "—"))
        extra.append(f"| {r['method']} | {rate_text} | {safe} / {total} | {systems} |")
    coverage = next((r.get("posterior_coverage") for r in parsed_rows
                     if r.get("posterior_coverage") not in (None, "")), "not recorded")
    extra.extend(["", f"Posterior coverage: {coverage}.", "",
        "Posterior MOCU is primary; safety rate and control magnitude are physical diagnostics.",
        "Safety requires both physical frequency and RoCoF limits to hold in the declared scenario.",
        "Rates average repeats within each physical system before averaging across systems.",
        "These are single-run estimates; across-seed standard deviations require multiple runs.",
        "Coverage is a decision preference, not a measured safety rate or engineering acceptance threshold."])
    return write_summary_md(
        exp_dir, system=system, experiment_type="objective_based", meta=meta,
        table_headers=["Method", "Terminal posterior MOCU"],
        table_rows=rows or [["(summary.csv missing)", "—"]], extra_lines=extra)


def write_eig_summary_md(
    exp_dir: Path,
    *,
    system: str,
    horizon: int,
    method_results: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Build eig_based table; primary metric is mean (terminal) EIG."""
    meta = dict(meta or {})
    meta.setdefault("T", horizon)
    meta.setdefault("N_obs", meta.get("n_obs", 0))
    if "observation_mode" not in meta:
        meta["observation_mode"] = observation_mode(int(meta["N_obs"]))

    headers = ["Method", "mean_eig", "mean_eig_step1", "sum_stepwise_eig", "n_rollouts"]
    rows: list[list[str]] = []
    for _key, payload in method_results.items():
        steps = list(payload.get("mean_eig_by_step") or [])
        mean_eig = payload.get("terminal_eig_mean")
        if mean_eig is None and steps:
            mean_eig = float(sum(steps))
        step1 = steps[0] if steps else None
        sum_steps = float(sum(steps)) if steps else None
        rows.append(
            [
                str(payload.get("method_label", _key)),
                _fmt(mean_eig, digits=4),
                _fmt(step1, digits=4),
                _fmt(sum_steps, digits=4),
                str(payload.get("n_rollouts", "—")),
            ]
        )

    return write_summary_md(
        exp_dir,
        system=system,
        experiment_type="eig_based",
        meta=meta,
        table_headers=headers,
        table_rows=rows,
    )
