"""Step-DAD for objective-based BOED.

Implements Hedman et al. (ICML 2025): start from trained DAD, infer the
posterior from a realized prefix, and fine-tune a private policy copy for the
remaining budget. Discrete designs use the paper's REINFORCE option.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.control.posterior_ctrl import normalize_log_weights
from src.objectives.mocu.context import (
    ExperimentContext, control_from_log_weights, observe_compressed,
    posterior_mocu, update_posterior_vector,
)
from src.objectives.mocu.train import _tensors_from_state, load_trained_policy


@dataclass(frozen=True)
class StepDADConfig:
    refinement_steps: int = 64
    fantasy_rollouts: int = 16
    learning_rate: float = 1.0e-4
    refine_from_step: int | None = None
    entropy_coefficient: float = 1.0e-3


def config_from_context(ctx: ExperimentContext, *, smoke: bool = False) -> StepDADConfig:
    raw = dict(ctx.cfg.training_for("objective_based") or {})
    refine = raw.get("step_dad_refine_from_step")
    return StepDADConfig(
        refinement_steps=2 if smoke else int(raw.get("step_dad_refinement_steps", 64)),
        fantasy_rollouts=4 if smoke else int(raw.get("step_dad_fantasy_rollouts", 16)),
        learning_rate=float(raw.get("step_dad_learning_rate", 1.0e-4)),
        refine_from_step=None if refine is None else int(refine),
        entropy_coefficient=float(raw.get("step_dad_entropy_coefficient", 1.0e-3)),
    )


def _refinement_step(ctx: ExperimentContext, cfg: StepDADConfig) -> int:
    if ctx.horizon < 2:
        return ctx.horizon
    step = max(1, ctx.horizon // 2) if cfg.refine_from_step is None else int(cfg.refine_from_step)
    return min(max(step, 1), ctx.horizon - 1)


def refine_policy(
    ctx: ExperimentContext,
    base_policy: torch.nn.Module,
    *,
    actions: list[int], observations: list[np.ndarray], log_w: np.ndarray,
    config: StepDADConfig, seed: int, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Fine-tune a private policy copy on posterior-conditioned rollouts."""
    policy = copy.deepcopy(base_policy).to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.learning_rate))
    rng = np.random.default_rng(int(seed))
    started = time.perf_counter()
    posterior = normalize_log_weights(np.asarray(log_w, dtype=np.float64))
    last_utility: float | None = None
    if int(ctx.horizon) <= len(actions) or int(config.refinement_steps) <= 0:
        policy.eval()
        return policy, {"seconds": 0.0, "updates": 0, "fantasy_rollouts": 0}

    for _update in range(int(config.refinement_steps)):
        returns: list[float] = []
        log_prob_sums: list[torch.Tensor] = []
        entropy_sums: list[torch.Tensor] = []
        for _fantasy in range(int(config.fantasy_rollouts)):
            particle = int(rng.choice(len(posterior), p=posterior))
            fa = list(actions)
            fy = [np.asarray(y).copy() for y in observations]
            fw = np.asarray(log_w, dtype=np.float64).copy()
            lps: list[torch.Tensor] = []
            ents: list[torch.Tensor] = []
            for step in range(len(fa), int(ctx.horizon)):
                tensors = _tensors_from_state(
                    ctx, actions=fa, observations=fy, log_w=fw,
                    step=step, device=device,
                )
                dist = policy.distribution(*tensors[:-1], tensors[-1])
                action_t = dist.sample()
                action = int(action_t.item())
                lps.append(dist.log_prob(action_t).reshape(()))
                ents.append(dist.entropy().reshape(()))
                clean = np.asarray(
                    ctx.centres_support[action, particle], dtype=np.float64
                )
                y = clean + float(ctx.sigma_y) * rng.normal(size=clean.shape)
                fa.append(action)
                fy.append(y)
                fw = update_posterior_vector(ctx, fw, action, y)
            returns.append(-float(posterior_mocu(ctx, fw)))
            log_prob_sums.append(torch.stack(lps).sum())
            entropy_sums.append(torch.stack(ents).sum())

        reward = torch.as_tensor(returns, dtype=torch.float32, device=device)
        advantage = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-6)
        loss = -(torch.stack(log_prob_sums) * advantage.detach()).mean()
        loss = loss - float(config.entropy_coefficient) * torch.stack(entropy_sums).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        last_utility = float(reward.mean().detach())

    policy.eval()
    return policy, {
        "seconds": float(time.perf_counter() - started),
        "updates": int(config.refinement_steps),
        "fantasy_rollouts": int(config.refinement_steps * config.fantasy_rollouts),
        "mean_final_training_utility": last_utility,
    }


def evaluate_step_dad(
    ctx: ExperimentContext, n_rollouts: int, *, eval_seed: int,
    config: StepDADConfig | None = None, device_name: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or config_from_context(ctx)
    chosen = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    device = torch.device("cpu" if chosen == "auto" else chosen)
    base = load_trained_policy(ctx, "DAD", device=device)
    refine_at = _refinement_step(ctx, cfg)
    rows: list[dict[str, Any]] = []
    refine_seconds = 0.0
    refine_fantasies = 0

    for rid in range(int(n_rollouts)):
        tid = rid % len(ctx.test_systems)
        system = ctx.test_systems[tid]
        policy = base
        actions: list[int] = []
        observations: list[np.ndarray] = []
        log_w = np.asarray(ctx.log_p0, dtype=np.float64).copy()
        ess = [float(1.0 / np.sum(normalize_log_weights(log_w) ** 2))]
        refined = False
        for step in range(int(ctx.horizon)):
            if not refined and step == refine_at:
                policy, report = refine_policy(
                    ctx, base, actions=actions, observations=observations,
                    log_w=log_w, config=cfg,
                    seed=int(eval_seed) + 100_003 * rid, device=device,
                )
                refine_seconds += float(report["seconds"])
                refine_fantasies += int(report["fantasy_rollouts"])
                refined = True
            tensors = _tensors_from_state(
                ctx, actions=actions, observations=observations,
                log_w=log_w, step=step, device=device,
            )
            with torch.no_grad():
                action = int(torch.argmax(policy(*tensors[:-1], tensors[-1]), dim=-1).item())
            y = observe_compressed(
                system, action, sigma_y=ctx.sigma_y, n_obs=ctx.n_obs,
                global_seed=int(eval_seed), theta_id=tid,
                rollout_id=rid, step=step,
            )
            actions.append(action)
            observations.append(y)
            log_w = update_posterior_vector(ctx, log_w, action, y)
            w = normalize_log_weights(log_w)
            ess.append(float(1.0 / np.sum(w**2)))
        rows.append({
            "method": "Step-DAD", "base_method": "step_dad",
            "eval_mode": "deterministic_semi_amortized",
            "rollout_id": rid, "theta_id": tid,
            "sequence": " ".join(map(str, actions)),
            "y_obs": str([np.asarray(y).tolist() for y in observations]),
            "ess_by_step": " ".join(f"{x:.4f}" for x in ess),
            "u_ctrl": float(control_from_log_weights(ctx, log_w).u_ctrl),
            "step_dad_refine_at": int(refine_at),
        })
    return rows, {
        "algorithm": "Step-DAD infer-refine (Hedman et al., ICML 2025)",
        "base_policy": "DAD", "refine_from_step": int(refine_at),
        "refinement_steps_per_instance": int(cfg.refinement_steps),
        "fantasy_rollouts_per_update": int(cfg.fantasy_rollouts),
        "refinement_seconds": float(refine_seconds),
        "refinement_fantasy_rollouts": int(refine_fantasies),
        "objective": "negative terminal safety-aware MOCU",
        "gradient_estimator": "REINFORCE for discrete designs",
    }


def step_dad_report(
    ctx: ExperimentContext, *, n_rollouts: int, seed: int, device: str,
    config: StepDADConfig | None = None, **_: Any,
) -> dict[str, Any]:
    rows, meta = evaluate_step_dad(
        ctx, n_rollouts=n_rollouts, eval_seed=seed,
        config=config, device_name=device,
    )
    return {"status": "complete", "n_rollouts": len(rows), **meta}
