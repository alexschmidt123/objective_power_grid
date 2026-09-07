"""Verify dependency wiring without contacting Slurm or launching work."""
from pathlib import Path
import io
import json
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from tools.submit_audit import extract_source, main


class SubmissionTests(unittest.TestCase):
    def test_archive_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'source.tar'
            with tarfile.open(archive,'w') as t:
                m=tarfile.TarInfo('../outside');m.size=1
                t.addfile(m,io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError,'Unsupported archive member'):
                extract_source(archive,root/'snapshot')

    def test_four_stage_submission_is_frozen_and_dependent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir();archive=root/'source.tar'
            names=['run.sh','sweep_run.sh','scripts/audit.sh','scripts/check.sh',
                   'hprc/master_audit.slurm','configs/ieee9_mocu.yaml']
            with tarfile.open(archive,'w') as t:
                for name in names:
                    m=tarfile.TarInfo(name);m.size=1;t.addfile(m,io.BytesIO(b'\n'))
            calls=[]
            def fake_sbatch(command,**kwargs):
                calls.append(command);return str(700+len(calls))+'\n'
            argv=['submit_audit','--archive',str(archive),'--commit','test-commit','--runtime',str(root)]
            with patch('sys.argv',argv),patch('tools.submit_audit.subprocess.check_output',side_effect=fake_sbatch):
                main()
            self.assertEqual(len(calls),4)
            for i in range(1,4):
                self.assertIn('--dependency=afterok:'+str(700+i),calls[i])
            out=next((root/'experiments').iterdir())
            self.assertTrue((out/'source_snapshot/data').is_symlink())
            manifest=json.loads((out/'source_manifest.json').read_text())
            self.assertEqual(manifest['commit'],'test-commit')
            self.assertIn('scripts/audit.sh',manifest['files'])
            self.assertFalse((out/'previous_campaign.json').exists())
