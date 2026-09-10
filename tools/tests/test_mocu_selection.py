import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src.objectives.mocu.train import checkpoint_score, evaluate_policy
from src.objectives.mocu.evaluate import rank_posterior_mocu

class SelectionTests(unittest.TestCase):
    def test_ranking_uses_posterior_not_regret_or_safety(self):
        rows=[dict(method='A', n=128, mean_posterior_mocu=.01, mean_mocu=.9, safety_rate=.5),
              dict(method='B', n=128, mean_posterior_mocu=.02, mean_mocu=.001, safety_rate=1),
              dict(method='C', n=128, mean_posterior_mocu=.01, safety_rate=.8)]
        ranked=rank_posterior_mocu(rows)
        self.assertEqual([r['method'] for r in ranked], ['A','C','B'])
        self.assertEqual([r['rank_by_posterior_mocu'] for r in ranked], [1,1,3])
        self.assertTrue(all(r['valid'] for r in rows))

    def test_nonfinite_is_not_ranked(self):
        rows=[dict(method='A', n=1, mean_posterior_mocu=float('nan'))]
        self.assertEqual(rank_posterior_mocu(rows), [])

    def test_checkpoint_validation_uses_posterior(self):
        trajectory=dict(actions=[0], terminal_u_ctrl=.1, terminal_posterior_mocu=.012)
        ctx=SimpleNamespace(undercontrol_penalty=10., violation_penalty=0.)
        with patch('src.objectives.mocu.train.sample_trajectory', return_value=trajectory):
            result=evaluate_policy(ctx,None,[{'u_req':.4}], n_rollouts=1,
                global_seed=1,reward_mode='dad_terminal',device='cpu')
        self.assertAlmostEqual(result['mean_mocu'], .012)
        self.assertAlmostEqual(result['mean_realized_regret'], 2.7)
        score,_=checkpoint_score(result['mean_posterior_mocu'],1,1,
                                diversity_weight=0,min_unique_fraction=0)
        self.assertAlmostEqual(score,.012)
