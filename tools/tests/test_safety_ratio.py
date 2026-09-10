import copy
import unittest
from pathlib import Path
import numpy as np
import torch
from src.config import SBOEDConfig, resolve_training_block
from src.objectives.mocu.train import _posterior_mocu_gpu


class PosteriorCoverageTests(unittest.TestCase):
    def test_quantile_minimizes_configured_loss(self):
        support = np.array([.1, .2, .3, .4, .5])
        weights = np.array([[.7, .12, .09, .06, .03], [.1, .2, .3, .2, .2]])
        for q in [.90, .95, .99]:
            cfg = SBOEDConfig({'control': {'posterior_coverage': q, 'alpha': .05},
                               'training': {'objective_based': {'undercontrol_penalty': 20}}}, Path('test.yaml'))
            tr = resolve_training_block(cfg.raw['training'], 'objective_based')
            penalty = tr['undercontrol_penalty']
            self.assertAlmostEqual(penalty, 1/(1-q))
            self.assertNotIn('min_valid_safety_rate', tr)
            costs = support[None, :] + penalty*np.maximum(support[:, None]-support[None, :], 0)
            exhaustive = weights @ costs
            mocu, action, _ = _posterior_mocu_gpu(
                torch.tensor(np.log(weights)), torch.tensor(support), torch.tensor(support),
                alpha=cfg.raw['control']['alpha'], margin=0, undercontrol_penalty=penalty,
                violation_penalty=0, robust_rule='quantile')
            np.testing.assert_allclose(mocu.numpy(), exhaustive.min(axis=1)-weights@support, atol=1e-12)
            np.testing.assert_allclose(action.numpy(), support[exhaustive.argmin(axis=1)])

    def test_legacy_unchanged(self):
        raw = {'control': {'alpha': .05}, 'training': {'undercontrol_penalty': 20}}
        before = copy.deepcopy(raw)
        SBOEDConfig(raw, Path('test.yaml'))
        self.assertEqual(raw, before)

    def test_invalid_ratio_rejected(self):
        for q in [0, 1, -.1, 1.1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                SBOEDConfig({'control': {'posterior_coverage': q}}, Path('test.yaml'))

    def test_incompatible_loss_rejected(self):
        for control, tr in [({'robust_rule': 'ibr_max'}, {}), ({'safety_margin': .1}, {}), ({}, {'violation_penalty': 1})]:
            with self.assertRaises(ValueError):
                SBOEDConfig({'control': dict(posterior_coverage=.9, **control), 'training': tr}, Path('test.yaml'))

    def test_eig_training_unchanged(self):
        eig = {'learning_rate': .01}
        cfg = SBOEDConfig({'control': {'posterior_coverage': .9}, 'training': {'eig_based': eig.copy()}}, Path('test.yaml'))
        self.assertEqual(cfg.raw['training']['eig_based'], eig)
