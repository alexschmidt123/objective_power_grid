"""Shared policy-state tensors, independent of any policy trainer."""
from __future__ import annotations
import numpy as np
import torch
from src.objectives.mocu.context import ExperimentContext, belief_summary


def _tensors_from_state(
    ctx: ExperimentContext,
    *,
    actions: list[int],
    observations: list[np.ndarray],
    log_w: np.ndarray,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    length = int(ctx.horizon)
    action_idx = torch.zeros(1, length, dtype=torch.long, device=device)
    obs = torch.zeros(1, length, ctx.obs_dim, dtype=torch.float32, device=device)
    mask = torch.zeros(1, length, dtype=torch.float32, device=device)
    if actions:
        n = len(actions)
        action_idx[0, :n] = torch.as_tensor(actions, dtype=torch.long)
        y = np.stack(
            [
                (np.asarray(o, dtype=np.float64) - ctx.obs_mean) / ctx.obs_std
                for o in observations
            ],
            axis=0,
        )
        obs[0, :n] = torch.as_tensor(y, dtype=torch.float32)
        mask[0, :n] = 1.0
    belief = torch.as_tensor(
        belief_summary(ctx, log_w, observations)[None, :],
        dtype=torch.float32,
        device=device,
    )
    steps = torch.as_tensor([step], dtype=torch.long, device=device)
    particles = torch.as_tensor(
        ctx.particle_features[None, :], dtype=torch.float32, device=device
    )
    weights = torch.as_tensor(
        np.exp(log_w - np.max(log_w))[None, :], dtype=torch.float32, device=device
    )
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    # Measurement times in SIR are chronological. Power-grid duration probes
    # are independent experiments and may be selected in any order; selected
    # durations are simply removed below by the non-chronological mask.
    chrono = str(getattr(ctx, "observation_mode", "")).startswith("sir_")
    if chrono:
        from src.domains.sir.design import chronological_feasible

        remaining = max(int(ctx.horizon) - int(step), 0)
        allowed = set(
            chronological_feasible(
                ctx.n_actions,
                list(actions),
                remaining_steps=remaining,
            ).tolist()
        )
        feasible = torch.zeros(1, ctx.n_actions, dtype=torch.bool, device=device)
        for a in allowed:
            feasible[0, int(a)] = True
    else:
        feasible = torch.ones(1, ctx.n_actions, dtype=torch.bool, device=device)
        for a in actions:
            feasible[0, int(a)] = False
    return action_idx, obs, mask, belief, steps, particles, weights, feasible
