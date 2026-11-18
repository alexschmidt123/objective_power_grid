"""Entry point for the isolated V4-Spec implementation."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/research/moe_sboed_v4_spec_20261118/run.py'),run_name='__main__')
