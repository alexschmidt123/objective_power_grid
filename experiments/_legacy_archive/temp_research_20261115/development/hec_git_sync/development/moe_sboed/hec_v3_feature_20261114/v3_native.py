"""Verified V2 runtime access; no dependency on retired V3 code."""
import importlib.util,pathlib,sys
sys.dont_write_bytecode=True
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
path=WORK/'development/moe_sboed/hec_v2_router_only_20261112/train_router.py'
spec=importlib.util.spec_from_file_location('hec_v3_feature_native_v2',path);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
