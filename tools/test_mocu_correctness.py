"""Small regression suite: terminal loss, baseline parity and bank publication."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import multiprocessing as mp
import tempfile
import time
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from src.banks.transaction import bank_lock, staged_bank
from src.control.posterior_ctrl import compute_u_ctrl, normalize_log_weights
from src.objectives.mocu.context import (
    posterior_mocu_batch, expected_mocu_after_action_vector, _score_fixed_subset,
    vector_gaussian_loglik,
)
from src.objectives.mocu.train import _posterior_mocu_gpu
from src.objectives.mocu.evaluate import summarize_rows


def concurrent_writer(destination):
    destination = Path(destination)
    with bank_lock(destination):
        if (destination / 'value.npy').exists():
            return
        with staged_bank(destination) as stage:
            with (destination.parent / 'generations').open('a') as f:
                f.write('generated\n')
            time.sleep(0.1)
            np.save(stage / 'value.npy', np.arange(8))


class Correctness(unittest.TestCase):
    def test_quantile_is_finite_loss_bayes_action_and_torch_matches(self):
        rng = np.random.default_rng(901)
        U = np.array([.31, .35, .35, .39, .41])
        grid = np.unique(np.r_[0., U, .5])
        log_w = rng.normal(0, 5, size=(25, len(U)))
        weights = np.array([normalize_log_weights(row) for row in log_w])
        for rule in ('quantile', 'ibr_max'):
            for snap in (True, False):
                loss, actions = posterior_mocu_batch(U, weights, alpha=.05, margin=0,
                    u_grid=grid, undercontrol_penalty=20, violation_penalty=0,
                    robust_rule=rule, snap_up=snap)
                if rule == 'quantile':
                    candidate_regrets = (grid[:, None] + 20*np.maximum(U-grid[:, None], 0)-U)
                    np.testing.assert_allclose(loss, (weights @ candidate_regrets.T).min(axis=1), atol=1e-12)
                for device in ('cpu', 'cuda'):
                    if device == 'cuda' and not torch.cuda.is_available():
                        continue
                    got, chosen, _ = _posterior_mocu_gpu(
                        torch.tensor(log_w, device=device), torch.tensor(U, device=device),
                        torch.tensor(grid, device=device), alpha=.05, margin=0,
                        undercontrol_penalty=20, violation_penalty=0,
                        robust_rule=rule, snap_up=snap)
                    np.testing.assert_allclose(got.cpu().numpy(), loss, atol=1e-12)
                    np.testing.assert_allclose(chosen.cpu().numpy(), actions, atol=1e-12)

    def test_quantile_changes_before_numerical_support_disappears(self):
        U = np.array([.35, .41]); w = np.array([.999999, .000001])
        self.assertEqual(compute_u_ctrl(U, w, alpha=.05, robust_rule='quantile'), .35)
        self.assertEqual(compute_u_ctrl(U, w, alpha=.05, robust_rule='ibr_max'), .41)

    def test_myopic_and_fixed_use_selected_loss(self):
        U = np.array([.35, .41]); prior = np.log([.5, .5])
        centres = np.array([[[0.], [3.]], [[0.], [0.]]])
        idx = np.array([0, 1, 0, 1]); noise = np.zeros((4, 1))
        common = dict(alpha=.05, margin=0., u_grid=U,
                      undercontrol_penalty=20., violation_penalty=0.)
        scores = {}
        for rule in ('quantile', 'ibr_max'):
            scores[rule] = []
            for action in (0, 1):
                y = centres[action, idx] + noise
                weights = np.array([normalize_log_weights(prior + vector_gaussian_loglik(
                    obs, centres[action], 1.)) for obs in y])
                expected, _ = posterior_mocu_batch(U, weights, robust_rule=rule, **common)
                got = expected_mocu_after_action_vector(action, prior, centres=centres,
                    U=U, sigma_y=1., idx=idx, noise=noise, robust_rule=rule, **common)
                self.assertAlmostEqual(got, expected.mean(), places=12)
                scores[rule].append(got)
            # Fixed uses a deterministic noise table; reconstruct independently.
            table = centres.transpose(1, 0, 2)
            epsilon = np.random.default_rng(17).normal(0, 1, size=(1, 2, 2, 1))
            losses = []
            for tid in range(2):
                w = normalize_log_weights(prior + vector_gaussian_loglik(
                    table[tid, 0]+epsilon[0, tid, 0], table[:, 0], 1.))
                losses.append(posterior_mocu_batch(U, w[None], robust_rule=rule, **common)[0][0])
            got = _score_fixed_subset((0,), centres_by_theta=table, U_support=U,
                log_p0=prior, sigma_y=1., seed=17, robust_rule=rule, **common)
            self.assertAlmostEqual(got, np.mean(losses), places=12)
        self.assertLess(scores['quantile'][0], scores['quantile'][1])
        self.assertAlmostEqual(scores['ibr_max'][0], scores['ibr_max'][1], places=12)

    def test_summary_separates_realized_and_posterior_estimators(self):
        row = dict(method='DAD', theta_id=0, u_ctrl=.4, control_gap=.1, ocu=.1,
                   posterior_mocu=.02, method_safe=1, u_ctrl_opt=.3, sequence='0 1')
        summary = summarize_rows([row], 'DAD')
        self.assertAlmostEqual(summary['mean_mocu'], .1)
        self.assertAlmostEqual(summary['mean_posterior_mocu'], .02)
        self.assertEqual(summary['metric_schema'], 'realized_operational_regret_v2')

    def test_oracle_enrichment_emits_realized_loss(self):
        from unittest.mock import patch
        from src.objectives.mocu.evaluate import attach_oracle
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(out_dir=Path(directory), test_systems=[{}],
                M_test=np.ones((1, 1)), K_test=np.ones((1, 1)), oracle_tolerance=1e-4,
                undercontrol_penalty=20., violation_penalty=0.)
            oracle = [{'theta_id': 0, 'u_ctrl_opt': .4, 'feasible': True}]
            with patch('src.objectives.mocu.evaluate.control_engine_for', return_value=(None, None)), \
                 patch('src.objectives.mocu.evaluate.load_or_compute_oracle_cache', return_value=oracle):
                rows, _ = attach_oracle(ctx, [{'method': 'DAD', 'theta_id': 0,
                    'u_ctrl': .39, 'posterior_mocu': .02}], skip_cuda_safety=True)
            self.assertAlmostEqual(rows[0]['ocu'], .19)
            self.assertAlmostEqual(rows[0]['posterior_mocu'], .02)

    def test_preflight_flags_constant_controls_without_using_test_systems(self):
        from src.objectives.mocu.preflight import decision_preflight
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(validation_systems=[{'obs_clean': np.zeros((2, 1)), 'u_req': .4}],
                log_p0=np.log([.5, .5]), n_actions=2, horizon=1, obs_dim=1,
                sigma_y=.1, centres_support=np.zeros((2, 2, 1)), U_support=np.array([.35, .41]),
                alpha=.05, margin=0., u_grid=np.array([.35, .41]), snap_up=True,
                robust_rule='quantile', undercontrol_penalty=20., violation_penalty=0., out_dir=directory)
            report = decision_preflight(ctx, rollouts=4)
            self.assertTrue(report['decision_degenerate'])
            self.assertFalse(report['physical_safety_certified'])

    def test_concurrent_bank_writers_generate_once(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'bank'
            ctx = mp.get_context('spawn')
            workers = [ctx.Process(target=concurrent_writer, args=(str(destination),)) for _ in range(2)]
            for worker in workers: worker.start()
            for worker in workers:
                worker.join(30)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual((Path(directory)/'generations').read_text().splitlines(), ['generated'])
            np.testing.assert_array_equal(np.load(destination/'value.npy'), np.arange(8))

    def test_failed_generation_preserves_existing_mmap(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'bank'; destination.mkdir()
            np.save(destination/'value.npy', np.arange(8))
            reader = np.load(destination/'value.npy', mmap_mode='r')
            with self.assertRaises(RuntimeError):
                with bank_lock(destination), staged_bank(destination) as stage:
                    np.save(stage/'value.npy', np.arange(2))
                    raise RuntimeError('validation rejected')
            np.testing.assert_array_equal(np.load(destination/'value.npy'), np.arange(8))
            with bank_lock(destination), staged_bank(destination) as stage:
                np.save(stage/'value.npy', np.arange(3))
            np.testing.assert_array_equal(reader, np.arange(8))
            np.testing.assert_array_equal(np.load(destination/'value.npy'), np.arange(3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
