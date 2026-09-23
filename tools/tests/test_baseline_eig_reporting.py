"""Check the statistical unit and incomplete conference seed handling."""
import math
import unittest
from tools.summarize_baseline_eig import summarize


class BaselineReportingTests(unittest.TestCase):
    def rows(self, method='random', count=5):
        return [dict(method=method, evaluation_seed=1001+i, system=j,
                     terminal_spce_nats=i+offset)
                for i in range(count) for j, offset in enumerate((-10, 10))]

    def test_five_seed_mean_and_sample_sd_for_both_baselines(self):
        for method in ('random', 'myopic'):
            result = summarize(self.rows(method), method)
            self.assertEqual(result['mean_spce_nats'], 2)
            self.assertAlmostEqual(result['sd_of_seed_means'], math.sqrt(2.5))
            self.assertEqual(result['n_evaluation_seeds'], 5)
            self.assertTrue(result['complete'])
            self.assertIn('+/-', result['display'])

    def test_three_seeds_cannot_claim_five(self):
        with self.assertRaisesRegex(ValueError, 'missing seeds'):
            summarize(self.rows(count=3), 'random')
        result = summarize(self.rows(count=3), 'random', allow_incomplete=True)
        self.assertEqual(result['missing_evaluation_seeds'], [1004, 1005])
        self.assertEqual(result['sd_of_seed_means'], 1)
        self.assertFalse(result['complete'])

    def test_shared_baselines_not_counted_as_independent_trainings(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize(self.rows()*3, 'random')

    def test_single_seed_sd_is_not_zero(self):
        result = summarize(self.rows(count=1), 'random', allow_incomplete=True)
        self.assertIsNone(result['sd_of_seed_means'])

    def test_invalid_or_unbalanced_evaluations(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, 'unequal'):
            summarize(rows[:-1], 'random')
        rows[0]['terminal_spce_nats'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            summarize(rows, 'random')


if __name__ == '__main__':
    unittest.main()
