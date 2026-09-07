"""Compatibility command implementation for the historical structural audit."""
import argparse
import json
from pathlib import Path

def cmd_bank_structure_audit(args: argparse.Namespace) -> None:
    """Legacy bank-structure audit, available only when explicitly invoked."""
    from src.experiment import (
        load_experiment_config, resolve_experiment_type, _resolve_exp_dir,
        ensure_result_layout,
    )
    from tools.audits.bank_structure import (
        run_bank_structure_audit,
        write_audit_report,
    )

    cfg = load_experiment_config(
        args.config,
        step_number=args.step_number,
        n_obs=args.n_obs,
        noise_sigma=args.noise_sigma,
    )
    exp_type = resolve_experiment_type(args.experiment_type)
    exp_dir = _resolve_exp_dir(
        cfg, exp_type, args.exp_dir, create_new=args.exp_dir is None
    )
    ensure_result_layout(exp_dir)
    bq = dict(cfg.raw.get("bank_quality") or {})
    report = run_bank_structure_audit(
        cfg,
        n_obs=int(args.n_obs),
        noise_sigma=float(args.noise_sigma),
        support_size=int(args.support_size),
        n_outer=int(args.n_outer),
        n_inner=int(args.n_inner),
        top_k=int(args.top_k),
        seed=int(args.seed),
        near_dup_corr=float(args.near_dup_corr),
        near_dup_frac_limit=float(args.near_dup_frac_limit),
        min_fixed_advantage=float(bq.get("min_fixed_advantage", 0.01)),
        min_distinct_second_actions=int(bq.get("min_distinct_second_actions", 2)),
        min_mean_branch_value=float(bq.get("min_mean_branch_value", 0.01)),
        max_mode_second_action_prob=float(bq.get("max_mode_second_action_prob", 0.75)),
        structure_audit_horizons=list(bq.get("structure_audit_horizons") or [2, 3, 4]),
        min_gap_improve_per_horizon=float(bq.get("min_gap_improve_per_horizon", 0.005)),
        max_fixed_subsets=int(bq.get("structure_audit_max_fixed_subsets", 220)),
    )
    out = Path(exp_dir) / "diagnostics"
    json_path, md_path = write_audit_report(report, out)
    print(json.dumps(report, indent=2))
    print(f"[bank-structure-audit] wrote {json_path}")
    print(f"[bank-structure-audit] wrote {md_path}")
    print(f"EXP_DIR={exp_dir}")
    # Explicit --bank-structure-audit always enforces trap / adaptive / monotone.
    # Normal runs skip these unless bank_quality.require_* is turned on in YAML.
    fails = []
    if not report.get("myopic_beatable"):
        fails.append("myopic_trap")
    if not report.get("adaptive_room"):
        fails.append("adaptive_room")
    if not report.get("monotone_adaptive_room"):
        fails.append("monotone_adaptive_room")
    if fails:
        raise SystemExit(
            "[bank-structure-audit] FAILED "
            f"{fails} — verdict={report.get('verdict')} "
            f"myopic_beatable={report.get('myopic_beatable')} "
            f"adaptive_room={report.get('adaptive_room')} "
            f"monotone_adaptive_room={report.get('monotone_adaptive_room')}. "
            "Pass/fail only (no data filtering). Retune YAML generation params "
            "(probe_durations, contingency, …) and regenerate with --force. "
            "After checks pass, omit --bank-structure-audit for result runs."
        )
