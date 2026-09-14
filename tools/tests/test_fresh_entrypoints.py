"""The public experiment launchers must reject cross-run reuse before work starts."""
import subprocess
import unittest
from pathlib import Path

class FreshEntrypointTests(unittest.TestCase):
    def test_both_launchers_reject_reuse(self):
        root=Path(__file__).resolve().parents[2]
        for script in ('run.sh','sweep_run.sh'):
            for flag in ('--reuse-fixed-run=/nonexistent','--resume=/nonexistent','--checkpoint=/nonexistent'):
                with self.subTest(script=script,flag=flag):
                    r=subprocess.run(['bash',str(root/script),flag],cwd=root,text=True,capture_output=True)
                    self.assertNotEqual(r.returncode,0)
                    self.assertIn('fresh experiments',r.stderr)

if __name__=='__main__':unittest.main()
