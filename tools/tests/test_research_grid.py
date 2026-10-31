"""GPU scientific checks for the new shared policies, independent of training."""
from pathlib import Path
import unittest
import numpy as np
import torch
import yaml
from src.research.benchmarks import make_benchmark
from src.research.policies import make_policy
from src.research.engine import generator,rollout,replay_likelihood


@unittest.skipUnless(torch.cuda.is_available(),'Requires the existing CUDA grid backend')
class GridResearchTests(unittest.TestCase):
    def test_carried_state_likelihood_and_full_horizon_gradient(self):
        root=Path(__file__).resolve().parents[2]
        torch.set_num_threads(1)
        for name in ['ieee9','ieee14']:
            config=root/'configs'/f'{name}_research_eig.yaml'
            raw=yaml.safe_load(config.read_text());raw['_config_path']=str(config)
            benchmark=make_benchmark(raw)
            for horizon in [3,5]:
                with self.subTest(network=name,horizon=horizon):
                    torch.manual_seed(1)
                    policy=make_policy(benchmark,horizon,'moe_sboed',8)
                    def objective():return rollout(benchmark,policy,horizon,2,45,3)['info'][:,-1].mean()
                    param=policy.experts[0][-1].bias
                    objective().backward();analytic=float(param.grad[0])
                    with torch.no_grad():
                        original=float(param[0]);param[0]=original+1e-4;plus=float(objective())
                        param[0]=original-1e-4;minus=float(objective());param[0]=original
                    np.testing.assert_allclose(analytic,(plus-minus)/2e-4,rtol=5e-3,atol=2e-3)
                    with torch.no_grad():
                        result=rollout(benchmark,policy,horizon,2,45,3)
                        lp,_,_=replay_likelihood(benchmark,result['particles'],result['actions'],result['observations'])
                    torch.testing.assert_close(lp,result['log_likelihood'][:,1:],rtol=1e-9,atol=1e-9)
                    actions=torch.stack(result['actions'],1)
                    self.assertTrue(torch.all(actions[:,1:]-actions[:,:-1]>=benchmark.gap-1e-9))
                    self.assertGreater(benchmark.physical_calls,benchmark.calls)


if __name__=='__main__':unittest.main()
