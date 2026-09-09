import unittest
import tempfile
import csv
from pathlib import Path
from src.objectives.mocu.evaluate import safety_statistics
from src.results.summary import write_objective_summary_md


class SafetyReportingTests(unittest.TestCase):
    def test_system_weighting_not_replicate_weighting(self):
        rows = [{'theta_id': 1, 'method_safe': 1}] + [
            {'theta_id': 2, 'method_safe': 0} for _ in range(3)]
        result = safety_statistics(rows)
        self.assertEqual(result['safety_rate'], .5)
        self.assertEqual(result['safety_n_systems'], 2)
        self.assertEqual(result['safety_n_outcomes'], 4)
        self.assertEqual(result['safety_safe_outcomes'], 1)

    def test_balanced_rate_matches_safe_fraction(self):
        rows = [{'theta_id': i, 'method_safe': s}
                for i, s in [(1, 1), (1, 0), (2, 1), (2, 1)]]
        self.assertEqual(safety_statistics(rows)['safety_rate'], .75)
        with self.assertRaises(ValueError):
            safety_statistics([{'theta_id': 1, 'method_safe': -1}])

    def test_report_uses_posterior_metric_and_keeps_unsafe_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'eval').mkdir()
            row = dict(method='DAD', mean_mocu=.9, mean_posterior_mocu=.012,
                       safety_rate=.75, valid=0, safety_safe_outcomes=3,
                       safety_n_outcomes=4, safety_n_systems=4,
                       posterior_coverage=.9)
            with (root/'eval/summary.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
            out = write_objective_summary_md(root, system='IEEE9').read_text()
            self.assertIn('75.00%', out)
            self.assertIn('0.012', out)
            self.assertIn('3 / 4', out)
            self.assertNotIn('INVALID', out)
