"""Regression tests for independent finite-loss space-audit planning."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch
from src.objectives.mocu.space_audit import AuditBudget, SpacePlanner, paired_interval


class AuditTests(unittest.TestCase):
    def test_uninformative_actions_have_zero_expected_value(self):
        planner=SpacePlanner(np.zeros((3,4,1)), [.35,.35,.41,.41], [.35,.41],
            sigma=1, budget=AuditBudget(inner=4, outer=3, first_candidates=3))
        w=planner.prior(2)
        used=torch.zeros((2,3), dtype=torch.bool)
        np.testing.assert_allclose(planner.one_step(w,used,17).numpy(), .03, atol=1e-12)
        seq,_=planner.fixed_sequence(2,49)
        self.assertEqual(len(set(seq)),2)
        rows=planner.evaluate(np.zeros((4,3,1)), [.35,.35,.41,.41], 2, seq, 901)
        np.testing.assert_allclose(rows['fixed']['loss'], rows['lookahead']['loss'])
        np.testing.assert_allclose(rows['myopic']['loss'], rows['lookahead']['loss'])
        for row in rows.values():
            self.assertTrue(all(len(set(seq))==2 for seq in row['sequence']))

    def test_chunking_does_not_change_candidate_fantasies(self):
        rng=np.random.default_rng(99)
        p=SpacePlanner(rng.normal(size=(5,8,2)), np.linspace(.35,.42,8), np.linspace(.35,.42,8),
            sigma=.5, budget=AuditBudget(inner=8, histories_per_batch=1, actions_per_batch=1))
        logw=p.tensor(rng.normal(size=(3,8)))
        used=torch.zeros((3,5),dtype=torch.bool)
        one=p.one_step(logw,used,91)
        p.budget.histories_per_batch=3; p.budget.actions_per_batch=5
        two=p.one_step(logw,used,91)
        np.testing.assert_allclose(one.numpy(),two.numpy(),atol=1e-12)

    def test_lookahead_can_find_complementary_probes(self):
        # XOR determines required control. Either bit alone leaves its posterior
        # distribution unchanged, whereas the pair identifies the required U.
        centres=np.array([[[0],[0],[0],[0]], [[-6],[-6],[6],[6]], [[-6],[6],[-6],[6]]])
        p=SpacePlanner(centres,[.35,.41,.41,.35],[.35,.41],sigma=1,
            budget=AuditBudget(inner=16,outer=16,first_candidates=3))
        action=p.choose(p.prior(),torch.zeros((1,3),dtype=torch.bool),depth=2,seed=177)
        self.assertIn(int(action[0]), [1,2])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cpu_cuda_planning_agreement(self):
        rng=np.random.default_rng(777)
        centres=rng.normal(size=(4,8,2))
        required=np.linspace(.35,.42,8)
        budget=AuditBudget(inner=16,outer=8,first_candidates=4)
        cpu=SpacePlanner(centres,required,required,sigma=.5,budget=budget)
        gpu=SpacePlanner(centres,required,required,sigma=.5,budget=budget,device='cuda')
        used=torch.zeros((2,4),dtype=torch.bool)
        np.testing.assert_allclose(cpu.one_step(cpu.prior(2),used,41).numpy(),
            gpu.one_step(gpu.prior(2),used.cuda(),41).cpu().numpy(),atol=1e-10)
        np.testing.assert_array_equal(cpu.choose(cpu.prior(2),used,depth=2,seed=41).numpy(),
            gpu.choose(gpu.prior(2),used.cuda(),depth=2,seed=41).cpu().numpy())

    def test_redundancy_diagnostics_detect_duplicate_probes(self):
        centres=np.tile(np.arange(4)[None,:,None],(5,1,1))
        p=SpacePlanner(centres,[.35,.35,.41,.41],[.35,.41],sigma=1)
        summary=p.structure_summary()
        self.assertAlmostEqual(summary['action_response_effective_rank'],1.)
        self.assertEqual(summary['fraction_action_pairs_cosine_above_0995'],1.)
        self.assertEqual(summary['informative_observation_coordinates'],1)

    def test_bootstrap_clusters_repeated_noise_seeds(self):
        differences=np.array([[.1,.2,.3,.4],[.1,.2,.3,.4]])
        result=paired_interval(differences,bootstrap=500)
        self.assertEqual(result['n_theta'],4)
        self.assertAlmostEqual(result['mean'],.25)
        repeated=paired_interval(np.repeat(differences,3,axis=0),bootstrap=500)
        np.testing.assert_allclose(result['ci95'],repeated['ci95'])


if __name__=='__main__':
    unittest.main(verbosity=2)
