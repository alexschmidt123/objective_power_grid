"""Write a single experiment-root ``summary.md`` comparison table.

Observation mode follows ``N_obs`` (max_rocof if 0, sampled Δf otherwise).
Primary metric: ``mean_cost_utility`` for cost_utility, ``mean_eig`` for eig_based.
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


def write_cost_utility_summary(exp_dir, summary, *, coverage, u_max):
    """Plain-text report for the continuous controller, without LaTeX."""
    lines = ["# Cost utility", "", "Utility = -u_ctrl / u_max. Higher is better.",
        f"u_max = {u_max:g}; required posterior joint safety probability = {coverage:g}.",
        "Safety requires BOTH the frequency nadir and maximum absolute RoCoF limits.",
        "Coverage is model-based; the safety rates below are independently evaluated.", "",
        "| Method | Mean utility | Mean control | Joint safety | Frequency safety | RoCoF safety |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("| " + " | ".join([r['method']] + [_fmt(r[k]) for k in
            ['mean_cost_utility','mean_u_ctrl','safety_rate','frequency_safety_rate','rocof_safety_rate']]) + " |")
    lines += ["", "No infeasible case is silently capped, dropped, or resampled.",
        "The terminal control window is checked; probe-period safety is not enforced by this implementation.",
        "The oracle uses each method's actual terminal state. It is a diagnostic, not subtracted from utility.", ""]
    path=Path(exp_dir)/'summary.md'
    path.write_text('\n'.join(lines))
    return path


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
