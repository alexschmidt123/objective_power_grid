#!/usr/bin/env python3
"""Numerical invariants for the primary Yoon IBR MOCU path."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control.posterior_ctrl import belief_mocu, ibr_max_u_ctrl
from src.objectives.mocu.train import _posterior_mocu_gpu


def main() -> None:
    rng = np.random.default_rng(20260904)
    u_optimal = np.sort(rng.uniform(0.1, 0.5, size=257)).astype(np.float64)
    u_grid = np.unique(np.concatenate(([0.0], u_optimal, [0.6])))

    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        for _ in range(32):
            log_w = rng.normal(size=(7, len(u_optimal))).astype(np.float64)
            w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            expected_u = np.asarray(
                [ibr_max_u_ctrl(u_optimal, row) for row in w], dtype=np.float64
            )
            expected_mocu = np.asarray(
                [belief_mocu(u_optimal, row, u) for row, u in zip(w, expected_u)],
                dtype=np.float64,
            )
            got_mocu, got_u, _ = _posterior_mocu_gpu(
                torch.as_tensor(log_w, dtype=torch.float64, device=device),
                torch.as_tensor(u_optimal, dtype=torch.float64, device=device),
                torch.as_tensor(u_grid, dtype=torch.float64, device=device),
                alpha=0.05,
                margin=0.0,
                undercontrol_penalty=20.0,
                violation_penalty=0.0,
                robust_rule="ibr_max",
            )
            np.testing.assert_allclose(got_u.cpu().numpy(), expected_u, atol=1e-12)
            np.testing.assert_allclose(
                got_mocu.cpu().numpy(), expected_mocu, rtol=1e-11, atol=1e-12
            )
            assert np.all(expected_mocu >= -1e-12)

    one_hot = np.zeros_like(u_optimal)
    one_hot[19] = 1.0
    u = ibr_max_u_ctrl(u_optimal, one_hot)
    assert abs(belief_mocu(u_optimal, one_hot, u)) < 1e-12
    equal = np.full(11, 0.35)
    weights = np.full(11, 1.0 / 11.0)
    assert abs(belief_mocu(equal, weights, 0.35)) < 1e-12
    print("MOCU_ALIGNMENT_OK")


if __name__ == "__main__":
    main()
