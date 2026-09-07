"""Shared catalog extraction must preserve complete bus coverage and order."""
import unittest
import numpy as np
from tools.audits.catalog import Catalog, candidate_sets, resolve_pool_actions


class CatalogTests(unittest.TestCase):
    def test_subset_retains_original_column_order(self):
        catalog=Catalog(((.1,2,.3),(.1,1,.3),(.1,1,.5),(.1,2,.5)))
        ids,groups=resolve_pool_actions(catalog,[.5,.3,.5])
        np.testing.assert_array_equal(ids,[0,1,2,3])
        np.testing.assert_array_equal(groups[.3],[0,1])

    def test_duplicate_bus_is_not_complete_coverage(self):
        catalog=Catalog(((.1,1,.3),(.1,1,.3),(.1,1,.5),(.1,2,.5)))
        with self.assertRaisesRegex(ValueError,'exactly once'):
            resolve_pool_actions(catalog,[.3])

    def test_sampled_candidates_preserve_baseline(self):
        baseline=(0,1,2,3,4,5)
        a=candidate_sets(list(range(12)),baseline,20,101)
        self.assertEqual(a,candidate_sets(list(range(12)),baseline,20,101))
        self.assertEqual(len(a),20)
        self.assertIn(baseline,a)
        self.assertTrue(all(len(set(row))==6 for row in a))
