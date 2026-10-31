import tempfile,unittest,torch
from pathlib import Path
from src.research.benchmarks import LocationFinding
from src.research.runner import DEFAULTS
from src.research.training import train_spce
from src.research.research_redq import train_rl_sboed
class Tests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1)
  self.b=LocationFinding({'research':{'paper_protocol':{'corrected_methods':True}}})
  self.s=DEFAULTS|dict(primary_episode_budget=1,updates=2,batch_size=4,contrasts=4,validate_every=100,hidden=16,rl_batch_size=4,rl_replay_batch=8,rl_replay_capacity=64,rl_min_buffer=2,rl_gradient_steps=2)
 def test_fixed_and_dad_exact_budget_not_update_ceiling(self):
  for kind in ['fixed','dad']:
   with tempfile.TemporaryDirectory() as d:
    _,r=train_spce(self.b,kind,3,self.s,101,Path(d),True,13)
    self.assertEqual(r['generated_training_trajectories'],13)
    self.assertEqual(r['contrast_training_trajectories'],52)
    self.assertEqual(r['actual_updates'],4)
 def test_rl_exact_budget_and_separate_contrasts(self):
  with tempfile.TemporaryDirectory() as d:
   _,r=train_rl_sboed(self.b,3,self.s,101,Path(d),13)
   self.assertEqual(r['generated_training_trajectories'],13)
   self.assertEqual(r['training_forward_trajectories'],65)
if __name__=='__main__':unittest.main()
