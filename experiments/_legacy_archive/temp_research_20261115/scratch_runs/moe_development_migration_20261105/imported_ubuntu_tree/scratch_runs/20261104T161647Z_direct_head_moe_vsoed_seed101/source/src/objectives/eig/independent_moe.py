"""Fresh, teacher-free MoE training for the discrete SIR EIG benchmark.

The objective matches the existing benchmark's posterior entropy reduction.
It is a finite-particle EIG approximation, not the online-grid sPCE estimator.
"""
from __future__ import annotations

import json
import time
import numpy as np
import torch
from torch import nn

from src.layout import model_dir
from src.policies.rl_sboed import PolicyConfig
from src.policies.independent_moe import ARCHITECTURE, IndependentMoEPolicy


def entropy(log_weights):
    weights = log_weights.softmax(-1)
    return -(weights * weights.clamp_min(1e-30).log()).sum(-1)


def state_inputs(ctx, actions, observations, step, log_weights, particles):
    batch, horizon = actions.shape
    mask = (torch.arange(horizon, device=actions.device)[None, :] < step).expand(batch, -1).float()
    normalized = (observations - float(ctx.obs_mean)) / float(ctx.obs_std)
    normalized = normalized * mask[..., None]
    stages = torch.full((batch,), step, device=actions.device, dtype=torch.long)
    weights = log_weights.softmax(-1)
    choices = torch.arange(ctx.n_actions, device=actions.device)[None, :]
    last = actions[:, step - 1:step] if step else -torch.ones_like(actions[:, :1])
    feasible = (choices > last) & (choices <= ctx.n_actions - (horizon - step))
    return (actions.clone(), normalized, mask, torch.zeros(batch, 33, device=actions.device),
            stages, particles.expand(batch, -1, -1), weights, feasible)


@torch.no_grad()
def batch_rollout(ctx, policy, clean, centres, particles, *, seed, stochastic,
                  rollout_ids=None):
    batch, horizon = len(clean), ctx.horizon
    device = clean.device
    actions = torch.zeros(batch, horizon, device=device, dtype=torch.long)
    observations = torch.zeros(batch, horizon, ctx.obs_dim, device=device)
    log_weights = torch.as_tensor(ctx.log_p0, device=device, dtype=torch.float32).expand(batch, -1).clone()
    h0 = entropy(log_weights)
    before = h0
    states, selected, old_lp, rewards = [], [], [], []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for stage in range(horizon):
        inputs = state_inputs(ctx, actions, observations, stage, log_weights, particles)
        dist = policy.distribution(*inputs)
        action = torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1) if stochastic else dist.logits.argmax(-1)
        if rollout_ids is None:
            noise = torch.randn((batch, ctx.obs_dim), generator=generator, device=device) * ctx.sigma_y
        else:
            # Exact existing evaluation noise convention, including validation IDs.
            noise = torch.as_tensor(np.stack([
                np.random.default_rng(int(seed) + 97451 * int(i) + 104729 * stage)
                .normal(0, ctx.sigma_y, size=(ctx.obs_dim,)).astype(np.float32)
                for i in rollout_ids]), device=device)
        y = clean[torch.arange(batch, device=device), action] + noise
        diff = centres[:, action, :].permute(1, 0, 2) - y[:, None, :]
        log_weights = log_weights - 0.5 * diff.square().sum(-1) / ctx.sigma_y**2
        after = entropy(log_weights)
        states.append(inputs)
        selected.append(action)
        old_lp.append(dist.log_prob(action))
        rewards.append(before - after)
        before = after
        actions[:, stage] = action
        observations[:, stage] = y
    return states, selected, old_lp, rewards, h0 - before, actions


def train_independent_moe(ctx, *, smoke, seed):
    if not str(ctx.observation_mode).startswith("sir_"):
        raise ValueError("Independent discrete MoE trainer currently supports SIR ODE only")
    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = ctx.cfg.training_for("eig_based")
    config = PolicyConfig(hidden=int(training.get("policy_hidden", 256)), max_steps=ctx.horizon,
                          obs_dim=ctx.obs_dim, particle_dim=ctx.particle_features.shape[1])
    variant = str(training.get("eig_moe_variant", "learned"))
    if variant not in {"learned", "uniform", "matched_dense"}:
        raise ValueError(f"Invalid independent MoE variant: {variant}")
    policy = IndependentMoEPolicy(ctx.n_actions, config,
                                 n_experts=int(training.get("eig_moe_n_experts", 4)),
                                 top_k=int(training.get("eig_moe_top_k", 2)),
                                 balance_coefficient=float(training.get("eig_moe_independent_balance", 0.01))).to(device)
    reference_parameters = sum(p.numel() for p in policy.parameters())
    if variant != "learned":
        # Reset RNG so the shared encoder receives the same initialization.
        torch.manual_seed(int(seed))
        n_experts = 1 if variant == "matched_dense" else policy.n_experts
        expert_hidden = config.hidden
        if variant == "matched_dense":
            # Each hidden head unit contributes hidden + n_actions + 1 weights.
            shared = reference_parameters - sum(p.numel() for p in policy.experts.parameters()) - sum(p.numel() for p in policy.router.parameters())
            expert_hidden = max(1, round((reference_parameters - shared - ctx.n_actions) / (config.hidden + ctx.n_actions + 1)))
        policy = IndependentMoEPolicy(ctx.n_actions, config, n_experts=n_experts,
                                     top_k=n_experts, expert_hidden=expert_hidden,
                                     routing="uniform", balance_coefficient=0.0).to(device)
    # Critic is a separately initialized value regressor, never a design policy.
    critic = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, 1)).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(training.get("learning_rate", .0003)))
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=float(training.get("eig_moe_critic_lr", .001)))
    train = torch.as_tensor(np.stack([s["obs_clean"] for s in ctx.train_systems]), dtype=torch.float32, device=device)
    n_val = 4 if smoke else int(training.get("eig_validation_systems", 256))
    val = torch.as_tensor(np.stack([s["obs_clean"] for s in ctx.validation_systems[:n_val]]), dtype=torch.float32, device=device)
    centres = torch.as_tensor(ctx.centres_support.transpose(1, 0, 2), dtype=torch.float32, device=device)
    particles = torch.as_tensor(ctx.particle_features, dtype=torch.float32, device=device)[None]
    if particles.shape[-1] == 3:
        particles = particles.clone()
        particles[..., 2] = 0
    epochs = 2 if smoke else int(training.get("eig_epochs", 200))
    per_epoch = 16 if smoke else int(training.get("eig_steps_per_epoch", 512))
    batch_size = 4 if smoke else int(training.get("batch_size", 256))
    ppo_epochs = 2 if smoke else int(training.get("eig_moe_ppo_epochs", 4))
    clip = float(training.get("eig_moe_ppo_clip", .2))
    entropy_coef = float(training.get("entropy_coef", .05))
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    history = []
    best_value, best_epoch, best_state = -float("inf"), 0, None
    for epoch in range(epochs + 1):
        mean_train, losses = [], []
        routing_stats = {}
        if epoch:
            policy.train()
            for batch_start in range(0, per_epoch, batch_size):
                count = min(batch_size, per_epoch - batch_start)
                ids = torch.as_tensor(rng.integers(len(train), size=count), device=device)
                states, actions, old_lps, rewards, gains, _ = batch_rollout(
                    ctx, policy, train[ids], centres, particles, seed=int(seed) + epoch * 1000003 + batch_start,
                    stochastic=True)
                inputs = tuple(torch.cat([s[i] for s in states], 0) for i in range(8))
                actions, old_lps = torch.cat(actions), torch.cat(old_lps)
                returns = torch.stack(rewards).flip(0).cumsum(0).flip(0).flatten()
                with torch.no_grad():
                    values = critic(policy.features(*inputs[:7])).squeeze(-1)
                    advantage = returns - values
                    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-8)
                for _ in range(ppo_epochs):
                    dist = policy.distribution(*inputs)
                    ratio = (dist.log_prob(actions) - old_lps).exp()
                    loss = -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).mean()
                    aux, routing_stats = policy.routing_loss(*inputs)
                    loss = loss - entropy_coef * dist.entropy().mean() + aux
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite independent MoE loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    # Detached actor features prevent value loss changing the policy.
                    with torch.no_grad():
                        features = policy.features(*inputs[:7])
                    value_loss = nn.functional.huber_loss(critic(features).squeeze(-1), returns)
                    critic_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optimizer.step()
                mean_train.extend(gains.cpu().tolist())
                losses.append(float(loss.detach()))
        policy.eval()
        *_, val_gains, val_actions = batch_rollout(
            ctx, policy, val, centres, particles, seed=seed, stochastic=False,
            rollout_ids=np.arange(len(val)) + 70000)
        value = float(val_gains.mean())
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite validation EIG")
        if value > best_value:
            best_value, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        row = {"epoch": epoch, "validation_terminal_eig": value,
               "validation_n_unique_sequences": len(torch.unique(val_actions, dim=0)),
               "mean_terminal_eig": float(np.mean(mean_train)) if mean_train else None,
               "mean_loss": float(np.mean(losses)) if losses else None,
               "elapsed_seconds": time.perf_counter() - started, **routing_stats}
        history.append(row)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[independent-moe] T={ctx.horizon} epoch={epoch}/{epochs} val={value:.5f} best={best_value:.5f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    policy.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    meta = {"architecture": ARCHITECTURE, "method": "moe_sboed", "training_seed": int(seed),
            "variant": variant, "routing": policy.routing,
            "reference_parameter_count": reference_parameters,
            "policy_hidden": config.hidden, "n_experts": policy.n_experts, "top_k": policy.top_k,
            "expert_hidden": policy.expert_hidden, "objective": "terminal_posterior_entropy_reduction",
            "estimator": "finite_particle_entropy_difference", "optimizer": "ppo_actor_critic",
            "teacher": None, "behavioral_cloning_trajectories": 0, "moe_step0_action": None,
            "first_action": "learned", "checkpoint_selection": "maximum_validation_eig_only",
            "best_stage": f"epoch_{best_epoch}", "best_validation_terminal_eig": best_value,
            "best_validation_n_unique_sequences": history[best_epoch]["validation_n_unique_sequences"],
            "epochs": epochs, "trajectories_per_epoch": per_epoch, "n_validation_systems": len(val),
            "parameter_count": sum(p.numel() for p in policy.parameters())}
    directory = model_dir(ctx.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "moe_sboed.pth"
    torch.save({"state_dict": best_state, "meta": meta, "elapsed_seconds": elapsed}, path)
    result = {**meta, "checkpoint": str(path), "elapsed_seconds": elapsed, "history": history}
    (directory / "moe_sboed_training.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
