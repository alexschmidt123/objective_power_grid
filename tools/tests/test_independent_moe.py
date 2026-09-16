"""Sparse execution, gradients, chronological rollout and teacher independence."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.policies.independent_moe import IndependentMoEPolicy, ARCHITECTURE
from src.policies.rl_sboed import PolicyConfig
from src.objectives.eig.independent_moe import batch_rollout, state_inputs, train_independent_moe
from src.objectives.eig.vector import VectorEIGEngine, _load_policy, _rollout


def context(directory, horizon=3):
    rng = np.random.default_rng(9)
    centres = rng.normal(size=(16, 8, 1)).astype(np.float32)
    training = {"policy_hidden": 16}
    return SimpleNamespace(
        horizon=horizon, n_actions=8, obs_dim=1, n_obs=1, sigma_y=1., obs_mean=0., obs_std=1.,
        observation_mode="sir_infected_count", particle_features=rng.normal(size=(16, 3)).astype(np.float32),
        log_p0=np.full(16, -np.log(16)), centres_support=centres.transpose(1, 0, 2),
        train_systems=[{"obs_clean": c} for c in centres],
        validation_systems=[{"obs_clean": c} for c in centres[:4]],
        out_dir=Path(directory), cfg=SimpleNamespace(training_for=lambda _: training))


class IndependentMoETests(unittest.TestCase):
    def test_cli_variant_reaches_nested_eig_training(self):
        from src.config import repo_root
        from src.experiment import load_experiment_config
        for variant in ("uniform", "matched_dense"):
            cfg = load_experiment_config(str(repo_root() / "configs/sir_ode_eig.yaml"),
                                         step_number=3, n_obs=1, noise_sigma=1., moe_variant=variant)
            self.assertEqual(cfg.training_for("eig_based")["eig_moe_variant"], variant)

    def test_uniform_mixture_is_exact_and_has_no_router(self):
        ctx = context(".")
        policy = IndependentMoEPolicy(8, PolicyConfig(hidden=16, max_steps=3), routing="uniform", top_k=4)
        inputs = state_inputs(ctx, torch.zeros(3, 3, dtype=torch.long), torch.zeros(3, 3, 1), 0,
                              torch.zeros(3, 16), torch.tensor(ctx.particle_features)[None])
        features = policy.features(*inputs[:7])
        expected = torch.stack([expert(features) for expert in policy.experts], 1).mean(1)
        expected = expected.masked_fill(~inputs[-1], -1e9)
        torch.testing.assert_close(policy(*inputs), expected)
        self.assertIsNone(policy.router)
        loss, _ = policy.routing_loss(*inputs)
        self.assertEqual(float(loss), 0.)

    def test_fresh_ablation_training_and_parameter_match(self):
        for variant in ("uniform", "matched_dense"):
            with tempfile.TemporaryDirectory() as directory:
                ctx = context(directory)
                ctx.cfg.training_for = lambda _: {"policy_hidden": 16, "eig_moe_variant": variant}
                result = train_independent_moe(ctx, smoke=True, seed=101)
                policy = _load_policy(ctx, "moe_sboed", torch.device("cpu"))
                self.assertIsNone(policy.router)
                self.assertEqual(result["variant"], variant)
                if variant == "matched_dense":
                    self.assertEqual(len(policy.experts), 1)
                    self.assertLess(abs(result["parameter_count"] - result["reference_parameter_count"]), 16+8+1)
                else:
                    self.assertEqual(len(policy.experts), 4)

    def test_sparse_execution_and_router_task_gradient(self):
        torch.manual_seed(4)
        ctx = context(".")
        policy = IndependentMoEPolicy(8, PolicyConfig(hidden=16, max_steps=3), n_experts=4, top_k=2)
        with torch.no_grad():
            policy.router.weight.zero_()
            policy.router.bias.copy_(torch.tensor([2., 1., -2., -3.]))
        inputs = state_inputs(ctx, torch.zeros(6, 3, dtype=torch.long), torch.zeros(6, 3, 1), 0,
                              torch.zeros(6, 16), torch.tensor(ctx.particle_features)[None])
        calls = [0] * 4
        handles = [expert.register_forward_hook(lambda m, x, y, e=e: calls.__setitem__(e, calls[e]+len(x[0])))
                   for e, expert in enumerate(policy.experts)]
        loss = -policy.distribution(*inputs).log_prob(torch.zeros(6, dtype=torch.long)).mean()
        loss.backward()
        self.assertEqual(calls, [6, 6, 0, 0])
        self.assertGreater(float(policy.router.bias.grad.abs().sum()), 0)
        for expert in policy.experts[:2]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in expert.parameters()), 0)
        for h in handles:
            h.remove()

    def test_batched_rollout_matches_scalar_likelihood(self):
        for horizon in (3, 4, 5):
            ctx = context(".", horizon)
            policy = IndependentMoEPolicy(8, PolicyConfig(hidden=16, max_steps=horizon))
            clean = torch.tensor(np.stack([s["obs_clean"] for s in ctx.validation_systems]))
            particles = torch.tensor(ctx.particle_features)[None]
            engine = VectorEIGEngine(ctx, torch.device("cpu"))
            *_, gains, actions = batch_rollout(ctx, policy, clean, engine.centres, particles,
                                               seed=1001, stochastic=False, rollout_ids=np.arange(4))
            self.assertTrue(bool((actions[:, 1:] > actions[:, :-1]).all()))
            for i in range(4):
                log_w = engine.log_p0.clone()
                from src.objectives.eig.vector import _observe
                for stage, action in enumerate(actions[i]):
                    y = _observe(ctx.validation_systems[i], int(action), sigma=1., rollout_id=i,
                                 step=stage, eval_seed=1001)
                    log_w = engine.update(log_w, int(action), torch.tensor(y))
                expected = engine.entropy(engine.log_p0) - engine.entropy(log_w)
                self.assertAlmostEqual(float(expected), float(gains[i]), places=5)

    def test_teacher_free_training_and_strict_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = context(directory)
            with patch.object(VectorEIGEngine, "action_scores", side_effect=AssertionError("teacher")), \
                 patch.object(VectorEIGEngine, "two_step_scores", side_effect=AssertionError("teacher")), \
                 patch("src.objectives.eig.vector._fixed_sequence", side_effect=AssertionError("fixed")), \
                 patch("src.objectives.eig.vector._prior_two_step_action", side_effect=AssertionError("prior teacher")):
                result = train_independent_moe(ctx, smoke=True, seed=101)
                policy = _load_policy(ctx, "moe_sboed", torch.device("cpu"))
            self.assertEqual(result["architecture"], ARCHITECTURE)
            self.assertEqual(result["behavioral_cloning_trajectories"], 0)
            self.assertIsNone(result["moe_step0_action"])
            self.assertIsInstance(policy, IndependentMoEPolicy)
            self.assertTrue(all(torch.isfinite(p).all() for p in policy.parameters()))


if __name__ == "__main__":
    unittest.main()
