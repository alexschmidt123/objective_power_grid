"""Decision-level tests: shared joint safety, score sign, and reporting."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.config import load_config, normalize_experiment_type
from src.control.continuous_control import CarriedStateControl, continuous_decision
from src.results.summary import write_cost_utility_summary

ROOT=Path(__file__).resolve().parents[2]

class CostUtilityTests(unittest.TestCase):
    def test_joint_safety_is_not_separate_marginal_coverage(self):
        c=object.__new__(CarriedStateControl)
        c.bounds=(0.,.5)
        c.spec=SimpleNamespace(rocof_limit_hz_s=1.,delta_f_nadir_hz=-.2)
        c.engine=SimpleNamespace(N=1,simulate_metrics_batch=lambda *a,**k:
            (np.array([.8,1.2,.8]),np.array([-.1,-.1,-.3])))
        _,_,safe=c.metrics(np.ones((3,2)),np.zeros((3,2)),[.2]*3)
        np.testing.assert_array_equal(safe,[True,False,False])
        # Frequency and RoCoF each pass for 2/3, but jointly only 1/3 pass.
        self.assertAlmostEqual(float(safe.mean()),1/3)

    def test_config_coverage_does_not_inject_a_loss_penalty(self):
        cfg=load_config(ROOT/'configs/ieee9_cost_utility.yaml')
        cfg.validate_cost_utility()
        self.assertNotIn('undercontrol_penalty',cfg.training_for('cost_utility'))
        cfg.raw['control']['posterior_coverage']=1.
        with self.assertRaises(ValueError):cfg.validate_cost_utility()

    def test_only_active_experiment_types_are_accepted(self):
        self.assertEqual(normalize_experiment_type('cost_utility'),'cost_utility')
        self.assertEqual(normalize_experiment_type('eig_based'),'eig_based')
        for old in ['mocu','msc','msc_based','objective_based']:
            with self.assertRaises(ValueError):normalize_experiment_type(old)

    def test_probability_requirement_selects_joint_quantile(self):
        low=continuous_decision([.1,.2,.4],[.1,.8,.1],.8,.5)
        high=continuous_decision([.1,.2,.4],[.1,.8,.1],.95,.5)
        self.assertEqual(low['u_ctrl'],.2)
        self.assertEqual(high['u_ctrl'],.4)
        self.assertGreater(low['cost_utility'],high['cost_utility'])

    def test_plain_text_report_keeps_unsafe_outcomes_visible(self):
        row=dict(method='dad',mean_cost_utility=-.4,mean_u_ctrl=.2,
            safety_rate=.5,frequency_safety_rate=.75,rocof_safety_rate=.75)
        with tempfile.TemporaryDirectory() as d:
            text=write_cost_utility_summary(d,[row],coverage=.9,u_max=.5).read_text()
        self.assertIn('-0.400000',text)
        self.assertIn('0.500000',text)
        self.assertIn('Higher is better',text)
        self.assertIn('probe-period safety is not enforced',text)

if __name__=='__main__':unittest.main()
