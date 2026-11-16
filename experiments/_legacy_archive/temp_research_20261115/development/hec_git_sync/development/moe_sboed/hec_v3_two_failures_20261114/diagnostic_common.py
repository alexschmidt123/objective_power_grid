"""Load the completed audit without shadowing the native specialization 'audit' module."""
import importlib.util,pathlib,sys
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('v3_two_failures_frozen_analysis',pathlib.Path(__file__).with_name('audit.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
__all__=['ROOT','WORK','CANON','CF','PILOT','n','d','np','torch','json','pathlib','time','load','LOCK','sha','signal','paired']
for name in __all__:globals()[name]=getattr(module,name)
