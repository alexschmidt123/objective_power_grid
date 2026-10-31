import unittest,pathlib,yaml
from src.research.runner import METHODS,ABLATIONS
from src.research.policies import make_policy
from src.research.benchmarks import LocationFinding
class Tests(unittest.TestCase):
 def test_no_active_old_moe(self):
  self.assertNotIn('moe_sboed',METHODS);self.assertEqual(ABLATIONS,[])
  with self.assertRaises(ValueError):make_policy(LocationFinding({}),30,'moe_sboed',64)
  root=pathlib.Path(__file__).resolve().parents[2]
  self.assertFalse((root/'src/research/moe_system.py').exists())
  for p in (root/'configs').glob('*.yaml'):
   d=yaml.safe_load(p.read_text())
   if isinstance(d,dict) and d.get('research',{}).get('enabled'):
    self.assertNotIn('moe_sboed',d['experiment']['methods'])
if __name__=='__main__':unittest.main()
