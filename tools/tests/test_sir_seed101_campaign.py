import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools import sir_seed101_campaign as c

class CampaignSafetyTests(unittest.TestCase):
 def test_native_dad_initialization_is_training_seed_controlled(self):
  import torch
  from src.research.journal_run import fresh_native_sir_dad
  torch.manual_seed(3);a=fresh_native_sir_dad(5,101)
  torch.manual_seed(999);b=fresh_native_sir_dad(5,101)
  c=fresh_native_sir_dad(5,102)
  self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()))
  self.assertTrue(any(not torch.equal(a.state_dict()[k],c.state_dict()[k]) for k in a.state_dict()))
 def test_fourhour_dispatch_uses_journal_budget_path(self):
  import yaml
  from src.research import runner
  with tempfile.TemporaryDirectory() as d:
   cfg=Path(d)/'sir.yaml'
   cfg.write_text(yaml.safe_dump({'system':{'name':'sir_ode'},'research':{'journal_protocol':{'version':'sir_seed101_native_fourhour_20261027_v1'}}}))
   with patch('src.research.journal_run.main') as dispatch:
    runner.main(['--config',str(cfg),'--journal','--methods','dad'])
    dispatch.assert_called_once()
    self.assertEqual(dispatch.call_args.args[0].methods,'dad')
 def test_exact_baseline_commands_and_no_moe(self):
  with tempfile.TemporaryDirectory() as d:
   campaign=Path(d)/'10272026_sir_seed101'
   for method in c.METHODS:
    argv=c.command(campaign,method)
    self.assertIn('--train-only',argv)
    self.assertEqual(argv[argv.index('--methods')+1],method)
    self.assertEqual(argv[argv.index('--seed')+1],'101')
   self.assertEqual(set(c.METHODS),{'random','fixed','dad','rl_sboed','vsoed','scorebed'})
 def test_changed_frozen_source_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'science.py').write_text('original')
   campaign=root/'campaign';campaign.mkdir();(campaign/'configs').mkdir()
   (campaign/'configs/dad.yaml').write_text('original config')
   manifest=dict(sha256={'science.py':c.sha(root/'science.py')},config_sha256={'dad':c.sha(campaign/'configs/dad.yaml')})
   c.write(campaign/'source_manifest.json',manifest)
   with patch.object(c,'ROOT',root):
    c.verify(campaign)
    (root/'science.py').write_text('changed')
    with self.assertRaisesRegex(RuntimeError,'Frozen source changed'):c.verify(campaign)
 def test_changed_effective_config_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);campaign=root/'campaign';(campaign/'configs').mkdir(parents=True)
   cfg=campaign/'configs/dad.yaml';cfg.write_text('original')
   c.write(campaign/'source_manifest.json',dict(sha256={},config_sha256={'dad':c.sha(cfg)}))
   cfg.write_text('tampered')
   with patch.object(c,'ROOT',root):
    with self.assertRaisesRegex(RuntimeError,'configuration changed'):c.verify(campaign)
 def test_atomic_status_is_valid_json(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'status.json'
   for i in range(20):
    c.write(p,dict(state='RUNNING',pid=123,iteration=i))
    self.assertEqual(json.loads(p.read_text())['iteration'],i)
   self.assertEqual(list(p.parent.glob('*.tmp')),[])

if __name__=='__main__':unittest.main()
