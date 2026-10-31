import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from tools.storage_policy import enforce_storage_policy

class StorageBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.temp = self.root / 'objective_power_grid_temp'
        self.stable = self.root / 'objective_power_grid'
        self.temp.mkdir(); self.stable.mkdir()

    def test_default_development_output(self):
        args = enforce_storage_policy(self.temp, [])
        self.assertEqual(args[0], '--output')
        self.assertIn(self.temp / 'scratch_runs', Path(args[1]).parents)

    def test_rejects_stable_write_and_parent_escape_from_temp(self):
        for output in ('../objective_power_grid/experiments/a', 'experiments/a', 'scratch_runs/../../escape'):
            with self.assertRaises(ValueError):
                enforce_storage_policy(self.temp, ['--output', output])

    def test_rejects_symlink_escape(self):
        (self.temp / 'scratch_runs').mkdir()
        (self.temp / 'scratch_runs/link').symlink_to(self.stable, target_is_directory=True)
        with self.assertRaises(ValueError):
            enforce_storage_policy(self.temp, ['--output=scratch_runs/link/run'])

    def test_stable_requires_frozen_authorization(self):
        with self.assertRaises(ValueError):
            enforce_storage_policy(self.stable, [])
        with self.assertRaises(ValueError):
            enforce_storage_policy(self.stable, ['--output', 'experiments/sir_ode/run'])

    def test_existing_run_collision(self):
        (self.temp / 'scratch_runs/run').mkdir(parents=True)
        with self.assertRaises(ValueError):
            enforce_storage_policy(self.temp, ['--output', 'scratch_runs/run'])

    def test_sweep_uses_scratch(self):
        args = enforce_storage_policy(self.temp, ['--sweep'])
        self.assertIn('--campaign-root', args)
        with self.assertRaises(ValueError):
            enforce_storage_policy(self.temp, ['--sweep', '--campaign-root', '../objective_power_grid/experiments'])

    def test_help_has_no_filesystem_side_effect(self):
        self.assertEqual(enforce_storage_policy(self.stable, ['--help']), ['--help'])

    def test_rejects_conflicting_and_abbreviated_output_flags(self):
        for args in (['--output', 'scratch_runs/good', '--output', '../objective_power_grid/experiments/bad'],
                     ['--output', 'scratch_runs/good', '--out', '../objective_power_grid/experiments/bad']):
            with self.assertRaises(ValueError):
                enforce_storage_policy(self.temp, args)

    def test_development_relative_output_becomes_absolute(self):
        args = enforce_storage_policy(self.temp, ['--output', 'scratch_runs/new_run'])
        self.assertEqual(args, ['--output', str(self.temp / 'scratch_runs/new_run')])

    def test_frozen_official_command_and_source_integrity(self):
        for rel in ('src/science.py', 'configs/model.yaml', 'run.sh', 'sweep_run.sh'):
            p = self.stable / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# frozen source\n')
        args = ['--output', 'experiments/location_finding/official_run']
        key = hashlib.sha256(json.dumps(args, separators=(',', ':')).encode()).hexdigest()
        auth = self.stable / 'reports/official_run_authorizations' / (key + '.json')
        auth.parent.mkdir(parents=True)
        record = {'status': 'VALIDATED_FROZEN_SOURCE', 'regression_status': 'PASS', 'argv': args,
                  'source_sha256': {str(p.relative_to(self.stable)): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.stable.rglob('*') if p.is_file()}}
        auth.write_text(json.dumps(record))
        self.assertEqual(enforce_storage_policy(self.stable, args), args)
        (self.stable / 'src/science.py').write_text('# changed source\n')
        with self.assertRaises(ValueError):
            enforce_storage_policy(self.stable, args)

if __name__ == '__main__':
    unittest.main()
