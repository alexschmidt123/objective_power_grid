import tempfile,unittest
from pathlib import Path
import torch
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.research_redq import RLPolicy,train_rl_sboed,incremental_rewards
from src.research.runner import DEFAULTS
class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(101)
    def test_reward_sum_matches_terminal_information(self):
        info=torch.tensor([[1.,-.5,3.],[2.,4.,5.]],dtype=torch.float64)
        torch.testing.assert_close(incremental_rewards(info).sum(-1),info[:,-1])
    def test_policy_cannot_access_truth_and_keeps_bounds(self):
        b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}});p=RLPolicy(b,3)
        results=[]
        for shift in [0.,100.]:
            with torch.no_grad():results.append(rollout(b,p,3,2,1001,0,truth=torch.zeros(2,4,dtype=torch.float64)+shift,noise=torch.zeros(2,3,1,dtype=torch.float64)))
        torch.testing.assert_close(results[0]['actions'][0],results[1]['actions'][0],rtol=0,atol=0)
        for result in results:
            self.assertTrue(all((a.abs()<=4).all() for a in result['actions']))
    def test_sample_only_gradients_budget_and_checkpoint(self):
        class Guard(LocationFinding):
            def forward(self,*args):
                if torch.is_grad_enabled():raise AssertionError('RL differentiated simulator')
                return super().forward(*args)
        b=Guard({'location_finding':{'design_bounds':[-4,4]}})
        s=DEFAULTS|dict(updates=2,contrasts=4,rl_batch_size=2,rl_replay_batch=8,rl_replay_capacity=64,rl_min_buffer=2,rl_gradient_steps=2,rl_log_every=1)
        with tempfile.TemporaryDirectory() as d:
            p,r=train_rl_sboed(b,3,s,101,Path(d),trajectory_budget=40)
            self.assertEqual(r['training_forward_trajectories'],40)
            self.assertGreater(r['gradient_updates'],0)
            self.assertEqual(r['target_entropy'],-2)
            restored=RLPolicy(b,3);restored.load_state_dict(torch.load(Path(d)/'rl_sboed.pt',weights_only=False)['state_dict'])
            f=torch.zeros(2,13,dtype=torch.float64)
            torch.testing.assert_close(p(f,0),restored(f,0),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
