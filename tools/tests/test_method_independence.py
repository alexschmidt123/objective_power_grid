"""Baseline execution must not require optional MoE implementations."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
BLOCKER = '''
import importlib.abc, sys
class BlockMoE(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if "moe" in fullname.lower():
            raise AssertionError("Baseline imported optional module: " + fullname)
sys.meta_path.insert(0, BlockMoE())
'''

class MethodIndependenceTests(unittest.TestCase):
    def run_isolated(self, code):
        result = subprocess.run([sys.executable, '-c', BLOCKER + textwrap.dedent(code)],
            cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_grid_preflight_selects_only_requested_optional_policies(self):
        self.run_isolated('''
            import tempfile, json
            from pathlib import Path
            from unittest.mock import patch
            from types import SimpleNamespace, ModuleType
            import numpy as np
            from src.objectives.eig import continuous_eig as runner
            # Orchestration test: numerical physics/training has separate tests.
            observer = SimpleNamespace(n_obs=5, bounds=(.2,3.), window=3.,
                                       times=np.linspace(0,3,5))
            cuda_stub=ModuleType('src.domains.swing.continuous_cuda')
            cuda_stub.CudaContinuousSwingObserver=lambda *a,**k: observer
            sys.modules[cuda_stub.__name__]=cuda_stub
            for method in ('dad','fixed','rl_sboed','step_dad','myopic','random'):
                with tempfile.TemporaryDirectory() as directory:
                    output = str(Path(directory)/'run')
                    argv = ['continuous_eig', '--config', 'configs/ieee9_eig.yaml',
                        '--methods', method, '--N_obs','5','--estimate-only',
                        '--output',output]
                    with patch.object(sys,'argv',argv), \\
                         patch.object(runner,'improve_objective'), \\
                         patch.object(runner.OnlineEIG,'rollout'), \\
                         patch.object(runner,'evaluate',return_value=[]), \\
                         patch.object(runner,'REDQTrainer') as redq:
                        runner.main()
                    assert redq.call_count == int(method == 'rl_sboed')
                    estimate=json.loads((Path(output)/'timing_estimate.json').read_text())
                    assert estimate['moe_training_update_seconds'] == 0
        ''')

    def test_sir_dense_training_and_checkpoint_loading_without_moe(self):
        self.run_isolated('''
            import tempfile
            from pathlib import Path
            from types import SimpleNamespace
            import numpy as np, torch
            from src.objectives.eig.vector import train_vector_eig_policy, _load_policy
            rng=np.random.default_rng(9)
            centres=rng.normal(size=(16,8,1)).astype(np.float32)
            training={'policy_hidden':16, 'eig_epochs':1, 'eig_steps_per_epoch':2,
                      'batch_size':2, 'eig_bc_trajectories':2}
            for method in ('dad_eig', 'rl_sboed_eig'):
                with tempfile.TemporaryDirectory() as directory:
                    ctx=SimpleNamespace(horizon=3,n_actions=8,obs_dim=1,n_obs=1,
                        M_support=np.linspace(1,2,16), K_support=np.linspace(2,3,16),
                        U_support=np.zeros(16), alpha=.9, margin=0., u_grid=np.array([0.,1.]),
                        sigma_y=1.,obs_mean=0.,obs_std=1.,observation_mode='sir_infected_count',
                        particle_features=rng.normal(size=(16,3)).astype(np.float32),
                        log_p0=np.full(16,-np.log(16)),centres_support=centres.transpose(1,0,2),
                        train_systems=[{'obs_clean':c} for c in centres],
                        validation_systems=[{'obs_clean':c} for c in centres[:4]],
                        out_dir=Path(directory),cfg=SimpleNamespace(training_for=lambda _:training))
                    train_vector_eig_policy(ctx,method=method,smoke=False,seed=101)
                    policy=_load_policy(ctx,method,torch.device('cpu'))
                    assert type(policy).__name__ == 'AdaptiveExperimentPolicy'
        ''')

if __name__ == '__main__':
    unittest.main()
