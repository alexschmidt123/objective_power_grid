import tempfile
import unittest
from pathlib import Path
from tools.collect_results import lightweight_manifest,copy_results

class CollectionTests(unittest.TestCase):
    def test_filters_preserves_names_verifies_and_rejects_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source';dest=Path(tmp)/'destination'
            rel='experiments/09142026/campaign'
            run=src/rel/'T3/run';(run/'models').mkdir(parents=True)
            (run/'run_config.json').write_text('{"hardware":{"execution_site":"hprc"}}')
            (run/'models/dad.pth').write_bytes(b'checkpoint')
            (run/'models/dad_training.json').write_text('[]')
            doc=lightweight_manifest(src,rel)
            self.assertEqual(len(doc['entries']),2)
            self.assertEqual(len(doc['excluded']),1)
            self.assertEqual(copy_results(src,dest,doc)['status'],'verified')
            self.assertEqual((dest/rel/'T3/run/run_config.json').read_bytes(),(run/'run_config.json').read_bytes())
            self.assertFalse((dest/rel/'T3/run/models/dad.pth').exists())
            (dest/rel/'T3/run/run_config.json').write_text('different')
            with self.assertRaises(ValueError):copy_results(src,dest,doc)
    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):lightweight_manifest(tmp,'experiments/09142026/../../outside')

if __name__=='__main__':unittest.main()
