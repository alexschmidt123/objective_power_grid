import unittest
from types import SimpleNamespace
import torch
from src.policies.direct_moe import DirectMoEPolicy
from src.objectives.eig.moe_optimization import make_optimizer


class MoEOptimizationTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(learning_rate=.001, moe_router_lr_scale=.25,
                      moe_lr_schedule='cosine', moe_lr_final_fraction=.1, updates=2000)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_groups_cover_all_parameters_once_and_decay(self):
        policy = DirectMoEPolicy(3, 1)
        opt, schedule = make_optimizer(policy, self.args())
        ids = [id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in policy.parameters()})
        self.assertEqual({id(p) for p in opt.param_groups[1]['params']},
                         {id(p) for p in policy.router.parameters()})
        previous = float('inf')
        for update in range(1, 2001):
            schedule(update)
            self.assertLessEqual(opt.param_groups[0]['lr'], previous)
            previous = opt.param_groups[0]['lr']
            self.assertAlmostEqual(opt.param_groups[1]['lr'] / previous, .25)
        self.assertAlmostEqual(previous, .0001)
        schedule(1)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], .001)

    def test_constant_equal_rates_match_existing_adam(self):
        torch.manual_seed(9)
        a = DirectMoEPolicy(3, 1)
        b = DirectMoEPolicy(3, 1)
        b.load_state_dict(a.state_dict())
        old = torch.optim.Adam(a.parameters(), lr=.001)
        new, schedule = make_optimizer(b, self.args(moe_router_lr_scale=1., moe_lr_schedule='constant'))
        x = torch.randn(8, 10)
        for update in range(1, 4):
            schedule(update)
            for model, optimizer in [(a, old), (b, new)]:
                optimizer.zero_grad()
                model(x, 1)[0].mean.square().mean().backward()
                optimizer.step()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
