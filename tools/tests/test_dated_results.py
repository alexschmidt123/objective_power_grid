"""Check date grouping without launching simulations or training."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.config import load_config
from src.layout import allocate_result_dir, find_latest_result_dir
from src.objectives.eig.continuous_eig import main
from src.online_cli import main as dispatch

ROOT=Path(__file__).resolve().parents[2]

class DatedResultsTests(unittest.TestCase):
    def test_allocator_and_discovery_use_flat_directory(self):
        cfg=load_config(ROOT/'configs/sir_ode_eig.yaml')
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            path=allocate_result_dir(cfg,'eig_based',project_root=base,stamp='09132026_120000')
            self.assertEqual(path.parent,base/'experiments')
            self.assertTrue(path.name.startswith('09132026-1200-sir_ode_eig_T'))
            second=allocate_result_dir(cfg,'eig_based',project_root=base,stamp='09132026_120000')
            self.assertIn('_r2_T',second.name)
            self.assertEqual(find_latest_result_dir(cfg,'eig_based',project_root=base),second)

    def test_online_default_path_is_dated_and_describes_observations(self):
        class StopBeforeSimulation(Exception):pass
        with patch('sys.argv',['run.sh','--config',str(ROOT/'configs/power_grid/ieee9_eig.yaml'),'--method','dad','--T','4']), patch('src.results.run_directory.allocate_run',side_effect=StopBeforeSimulation) as allocate:
            with self.assertRaises(StopBeforeSimulation):
                main()
        allocate.assert_called_once_with(ROOT/'experiments','ieee9_eig',4,['dad'])

    def test_sweep_delegates_each_cell_to_dated_run_default(self):
        with patch('sys.argv',['sweep_run.sh','--sweep','--configs','ieee9_research_eig','--T','3,4','--objective','eig']),patch('src.online_cli.subprocess.run') as run:
            dispatch()
        self.assertEqual(run.call_count,2)
        for call in run.call_args_list:
            cmd=call.args[0]
            self.assertEqual(Path(cmd[1]).name,'run.sh')
            self.assertNotIn('--output',cmd)

if __name__=='__main__':unittest.main()
