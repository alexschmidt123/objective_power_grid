#!/usr/bin/env python3
"""Regenerate only the MOCU extension of an existing power-grid probe bank.

The EIG/probe arrays (theta_M, theta_K, delta_f, max_rocof) are read-only.
For each identical theta row this writes the minimum safe terminal control,
the full candidate safety table, and the per-theta OCU table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.banks.power_grid import resolve_dataset_dir
from src.config import load_config_for_run
from src.control.cuda_control import CudaControlEngine
from src.control.u_req import ControlSpec
from src.domains.swing.design import build_simulator


def _canonical_spec(spec: ControlSpec) -> dict[str, Any]:
    return {
        "control_model": "supplementary_active_power_injection",
        "alpha": float(spec.alpha),
        "robust_rule": str(spec.robust_rule),
        "safety_margin": float(spec.safety_margin),
        "snap_up": bool(spec.snap_up),
        "rocof_limit_hz_s": float(spec.rocof_limit_hz_s),
        "delta_f_nadir_hz": float(spec.delta_f_nadir_hz),
        "profile": {
            "bus": int(spec.profile.bus),
            "t_start": float(spec.profile.t_start),
            "duration": float(spec.profile.duration),
            "shape": str(spec.profile.shape),
            "units": str(spec.profile.units),
        },
        "contingency": {
            "bus": int(spec.contingency.bus),
            "magnitude": float(spec.contingency.magnitude),
            "units": str(spec.contingency.units),
        },
        "u_candidates": [float(x) for x in spec.u_candidates],
        "T_obs_sec": float(spec.T_obs_sec),
        "ode_dt": float(spec.ode_dt),
        "fs_hz": float(spec.fs_hz),
    }


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_save(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def _simulate_metrics_torch(
    sim: Any,
    spec: ControlSpec,
    M_rows: np.ndarray,
    K_rows: np.ndarray,
    u_mags: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Torch-CUDA implementation of the control-only RK4 kernel.

    This mirrors ``src/control/cuda_control.py`` and avoids a runtime nvcc
    dependency on managed machines that provide the CUDA runtime but not the
    compiler headers.
    """
    import math
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable and PyCUDA compilation failed")
    device = torch.device("cuda")
    dtype = torch.float64
    B = torch.as_tensor(sim.B, dtype=dtype, device=device)
    P_base = torch.as_tensor(sim.P_m, dtype=dtype, device=device)
    D = torch.as_tensor(sim.D_nodes, dtype=dtype, device=device)
    theta0 = torch.as_tensor(sim.theta0, dtype=dtype, device=device)
    omega0 = torch.as_tensor(sim.omega0, dtype=dtype, device=device)
    dt = float(spec.ode_dt)
    n_steps = int(math.ceil(float(spec.T_obs_sec) / dt))
    prof = spec.profile
    cont = spec.contingency
    n_theta = int(len(M_rows))
    n_u = int(len(u_mags))
    n_total = n_theta * n_u
    out_rocof = np.empty(n_total, dtype=np.float64)
    out_nadir = np.empty(n_total, dtype=np.float64)
    M_all = torch.as_tensor(M_rows, dtype=dtype, device=device)
    K_all = torch.as_tensor(K_rows, dtype=dtype, device=device)
    U_all = torch.as_tensor(u_mags, dtype=dtype, device=device)

    def profile_value(t: float, U: Any) -> Any:
        if prof.duration <= 0.0 or t < prof.t_start or t > prof.t_start + prof.duration:
            return torch.zeros_like(U)
        tau = (t - prof.t_start) / prof.duration
        if prof.shape == "step":
            return U
        if prof.shape == "hann":
            return U * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau))
        return U * tau

    for start in range(0, n_total, int(batch_size)):
        end = min(n_total, start + int(batch_size))
        flat = torch.arange(start, end, dtype=torch.long, device=device)
        theta_index = torch.div(flat, n_u, rounding_mode="floor")
        control_index = torch.remainder(flat, n_u)
        M = M_all[theta_index]
        K = K_all[theta_index]
        U = U_all[control_index]
        theta = theta0.expand(end - start, -1).clone()
        omega = omega0.expand(end - start, -1).clone()
        previous = omega.clone()
        rocof_max = torch.zeros(end - start, dtype=dtype, device=device)
        nadir = torch.full((end - start,), float("inf"), dtype=dtype, device=device)
        decay = K / (2.0 * math.pi) + D

        def rhs(th: Any, om: Any, injection: Any) -> tuple[Any, Any]:
            diff = th[:, :, None] - th[:, None, :]
            coupling = (B[None, :, :] * torch.sin(diff)).sum(dim=2)
            power = P_base.expand_as(om).clone()
            power[:, int(cont.bus)] += float(cont.magnitude)
            power[:, int(prof.bus)] += injection
            return om, (power - coupling - decay * om) / M

        for step in range(n_steps):
            t = step * dt
            z = profile_value(t, U)
            zm = profile_value(t + 0.5 * dt, U)
            ze = profile_value(t + dt, U)
            k1t, k1o = rhs(theta, omega, z)
            k2t, k2o = rhs(theta + 0.5 * dt * k1t, omega + 0.5 * dt * k1o, zm)
            k3t, k3o = rhs(theta + 0.5 * dt * k2t, omega + 0.5 * dt * k2o, zm)
            k4t, k4o = rhs(theta + dt * k3t, omega + dt * k3o, ze)
            theta = theta + (dt / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
            omega = omega + (dt / 6.0) * (k1o + 2.0 * k2o + 2.0 * k3o + k4o)
            delta_f = omega / (2.0 * math.pi)
            nadir = torch.minimum(nadir, delta_f.min(dim=1).values)
            rocof_step = ((omega - previous) / (2.0 * math.pi) / dt).abs().max(dim=1).values
            rocof_max = torch.maximum(rocof_max, rocof_step)
            previous = omega.clone()
        out_rocof[start:end] = rocof_max.cpu().numpy()
        out_nadir[start:end] = nadir.cpu().numpy()
    return out_rocof, out_nadir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh psi_star and OCU tables without touching the EIG probe bank"
    )
    parser.add_argument("--config", default="configs/ieee9_mocu.yaml")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        default=None,
        help="Separate MOCU-bank directory (probe arrays remain in the EIG bank)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even when a complete matching control bank exists",
    )
    args = parser.parse_args()

    cfg = load_config_for_run(args.config, _ROOT, step_number=3)
    probe_data_dir = resolve_dataset_dir(cfg, _ROOT)
    output = args.output or (cfg.raw.get("data") or {}).get("mocu_dataset_dir")
    if not output:
        raise RuntimeError("No --output and no data.mocu_dataset_dir in config")
    data_dir = Path(str(output))
    if not data_dir.is_absolute():
        data_dir = _ROOT / data_dir
    data_dir = data_dir.resolve()
    (data_dir / "meta").mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        (data_dir / split).mkdir(parents=True, exist_ok=True)
    spec = ControlSpec.from_cfg(cfg)
    spec_doc = _canonical_spec(spec)
    spec_hash = _digest(spec_doc)

    required = [
        *(data_dir / split / name for split in ("train", "test") for name in (
            "theta_M.npy", "theta_K.npy", "psi_star.npy", "control_safe.npy",
            "control_rocof.npy", "control_nadir.npy",
        )),
        data_dir / "meta" / "control_bank.yaml",
    ]
    meta_path = data_dir / "meta" / "control_bank.yaml"
    existing_meta = (
        yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        if meta_path.is_file() else {}
    )
    if (
        not args.force
        and all(path.is_file() for path in required)
        and existing_meta.get("control_spec_sha256") == spec_hash
        and Path(str(existing_meta.get("source_probe_bank", ""))).resolve()
        == probe_data_dir.resolve()
    ):
        print(f"[mocu-bank] complete and matched -> {data_dir}; skipping generation")
        return

    sim = build_simulator(cfg)
    sim.T_obs_sec = float(spec.T_obs_sec)
    sim.ode_dt = float(spec.ode_dt)
    sim.fs_hz = float(spec.fs_hz)
    candidates = spec.u_grid()
    reports: dict[str, Any] = {}

    for split in ("train", "test"):
        source_split_dir = probe_data_dir / split
        split_dir = data_dir / split
        m_path = source_split_dir / "theta_M.npy"
        k_path = source_split_dir / "theta_K.npy"
        probe_path = source_split_dir / "delta_f.npy"
        M = np.asarray(np.load(m_path), dtype=np.float64)
        K = np.asarray(np.load(k_path), dtype=np.float64)
        if M.shape != K.shape or M.ndim != 2:
            raise RuntimeError(f"{split}: incompatible theta shapes M={M.shape} K={K.shape}")
        probe_shape = np.load(probe_path, mmap_mode="r").shape
        if int(probe_shape[0]) != int(M.shape[0]):
            raise RuntimeError(
                f"{split}: theta/probe row mismatch {M.shape[0]} != {probe_shape[0]}"
            )

        n, n_nodes = M.shape
        n_u = int(candidates.size)
        print(f"[{split}] {n} theta x {n_u} controls = {n*n_u} control simulations")
        rocof, nadir = _simulate_metrics_torch(
            sim, spec, M, K, candidates, batch_size=int(args.batch_size)
        )
        rocof = rocof.reshape(n, n_u)
        nadir = nadir.reshape(n, n_u)
        safe = (rocof <= spec.rocof_limit_hz_s) & (
            nadir >= spec.delta_f_nadir_hz
        )
        feasible = safe.any(axis=1)
        monotone = ~np.any(safe[:, :-1] & ~safe[:, 1:], axis=1)
        uncontrolled_safe = safe[:, 0] if abs(float(candidates[0])) <= 1e-12 else None
        if (
            not feasible.all()
            or not safe[:, -1].all()
            or not monotone.all()
            or uncontrolled_safe is None
            or uncontrolled_safe.any()
        ):
            raise RuntimeError(
                f"{split}: invalid control bank: infeasible={int((~feasible).sum())}, "
                f"u_max_unsafe={int((~safe[:, -1]).sum())}, "
                f"nonmonotone={int((~monotone).sum())}, "
                f"u_zero_missing={int(uncontrolled_safe is None)}, "
                f"u_zero_safe={int(uncontrolled_safe.sum()) if uncontrolled_safe is not None else -1}"
            )

        first_safe = np.argmax(safe, axis=1)
        psi_star = candidates[first_safe]
        _atomic_save(split_dir / "psi_star.npy", psi_star.astype(np.float64))
        _atomic_save(split_dir / "u_optimal.npy", psi_star.astype(np.float64))
        _atomic_save(split_dir / "theta_M.npy", M.astype(np.float64))
        _atomic_save(split_dir / "theta_K.npy", K.astype(np.float64))
        _atomic_save(split_dir / "control_safe.npy", safe.astype(np.bool_))
        _atomic_save(split_dir / "control_rocof.npy", rocof.astype(np.float64))
        _atomic_save(split_dir / "control_nadir.npy", nadir.astype(np.float64))

        reports[split] = {
            "n_theta": int(n),
            "theta_dimension": int(2 * n_nodes),
            "n_control_candidates": int(n_u),
            "n_control_simulations": int(n * n_u),
            "psi_star_shape": [int(x) for x in psi_star.shape],
            "psi_star_min": float(psi_star.min()),
            "psi_star_max": float(psi_star.max()),
            "psi_star_mean": float(psi_star.mean()),
            "psi_star_std": float(psi_star.std()),
            "psi_star_unique": int(np.unique(psi_star).size),
            "particle_safety_rate": float(safe[np.arange(n), first_safe].mean()),
            "u_max_safety_rate": float(safe[:, -1].mean()),
            "theta_M_sha256": _file_digest(m_path),
            "theta_K_sha256": _file_digest(k_path),
            "probe_delta_f_sha256": _file_digest(probe_path),
        }

    metadata = {
        "schema": "mocu_control_extension_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_probe_bank": str(probe_data_dir.resolve()),
        "observation_storage": (
            "delta_f is not duplicated; use source_probe_bank/{split}/delta_f.npy "
            "with identical row indices"
        ),
        "source_relationship": (
            "row n shares the identical theta across theta_M, theta_K, delta_f, "
            "u_optimal and control tables"
        ),
        "control_spec_sha256": spec_hash,
        "control_spec": spec_doc,
        "definitions": {
            "psi_star[n]": "u_optimal: smallest safe candidate control for theta_n",
            "u_optimal[n]": "smallest safe candidate control for theta_n",
            "control_safe[n,j]": "physical safety of candidate u_j for theta_n",
            "control_rocof[n,j]": "maximum absolute ROCOF [Hz/s] over control horizon",
            "control_nadir[n,j]": "minimum frequency deviation [Hz] over control horizon",
        },
        "splits": reports,
    }
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_meta.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    os.replace(tmp_meta, meta_path)
    print(f"MOCU control extension regenerated -> {data_dir}")
    print(f"control_spec_sha256={spec_hash}")


if __name__ == "__main__":
    main()
