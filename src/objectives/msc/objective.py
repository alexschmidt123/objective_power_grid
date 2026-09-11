"""Discrete frequency-security-constrained terminal control."""
import numpy as np
from src.control.posterior_ctrl import posterior_control_decision


def validate_msc_support(required, grid, safe=None):
    required = np.asarray(required, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all() or np.any(grid < 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("MSC requires a finite, nonnegative, increasing admissible control grid")
    if required.size == 0 or not np.isfinite(required).all() or np.any(required < 0):
        raise ValueError("MSC requires finite feasible minimum-control requirements; do not drop infeasible systems")
    if grid.size == 0 or np.any(required > grid[-1] + 1e-12):
        raise ValueError("MSC infeasible: a bank requirement exceeds the available control grid")
    if safe is not None:
        safe = np.asarray(safe, dtype=bool)
        expected = grid[None, :] + 1e-12 >= required[:, None]
        if safe.shape != expected.shape or not np.array_equal(safe, expected):
            raise ValueError("MSC scalar requirement must match every stored grid safety condition; monotone safety is required")


def posterior_msc(required, weights, *, coverage, grid):
    """Smallest grid control with posterior safe mass >= coverage; no loss penalty."""
    if not 0 < float(coverage) < 1:
        raise ValueError("coverage must lie in (0, 1)")
    validate_msc_support(required, grid)
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid posterior weights")
    return posterior_control_decision(required, weights, 1-float(coverage),
        u_grid=grid, snap_up=True, robust_rule="quantile").u_ctrl
