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
    def test_legacy_allocator_and_discovery_use_date_layer(self):
        cfg=load_config(ROOT/'configs/sir_ode_eig.yaml')
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            path=allocate_result_dir(cfg,'eig_based',project_root=base,stamp='09132026_120000')
            self.assertEqual(path.parent,base/'experiments/09132026')
            self.assertEqual(find_latest_result_dir(cfg,'eig_based',project_root=base),path)

    def test_online_default_path_is_dated_and_describes_observations(self):
        class StopBeforeSimulation(Exception):pass
        captured=[]
        def intercept(path,*args,**kwargs):
            captured.append(path);raise StopBeforeSimulation()
        with patch('sys.argv',['run.sh','--config',str(ROOT/'configs/ieee9_eig.yaml'),'--objective','eig']),patch.object(Path,'mkdir',intercept):
            with self.assertRaises(StopBeforeSimulation):main()
        self.assertEqual(captured[0].parent.name,datetime.now().strftime('%m%d%Y'))
        self.assertTrue(captured[0].name.startswith(captured[0].parent.name+'_'))
        self.assertIn('_eig_T3_Nobs0_sigma0p005_W3_',captured[0].name)

    def test_sweep_delegates_each_cell_to_dated_run_default(self):
        with patch('sys.argv',['sweep_run.sh','--sweep','--T','3,4','--objective','eig']),patch('src.online_cli.subprocess.run') as run:
            dispatch()
        self.assertEqual(run.call_count,2)
        for call in run.call_args_list:
            cmd=call.args[0]
            self.assertEqual(Path(cmd[1]).name,'run.sh')
            self.assertNotIn('--output',cmd)

if __name__=='__main__':unittest.main()
