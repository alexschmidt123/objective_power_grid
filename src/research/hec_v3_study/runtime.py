"""Read-only historical relocation: path literals only; scientific source preserved."""
import contextlib,fcntl,hashlib,importlib.machinery,importlib.util,io,json,os,pathlib,sys,time
sys.dont_write_bytecode=True
MAIN=pathlib.Path.home()/'Documents/objective_power_grid';ARCHIVE=MAIN/'experiments/_legacy_archive/temp_research_20261115'
HERE=pathlib.Path(__file__).resolve().parent;REPORT=MAIN/'reports/hec_v3_study_20261115';OUT=MAIN/'experiments/location_finding/hec_v3_study_20261115'
OLD='Documents/objective_power_grid_temp';NEW='Documents/objective_power_grid/experiments/_legacy_archive/temp_research_20261115'
_RELOCATED={};_get_code=importlib.machinery.SourceFileLoader.get_code

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def put(path,value):
 path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def relocated_code(loader,fullname):
 path=pathlib.Path(loader.path)
 if path.is_relative_to(ARCHIVE):
  source=path.read_text();changed=source.replace(OLD,NEW)
  _RELOCATED[str(path)]=dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),executed_sha256=hashlib.sha256(changed.encode()).hexdigest(),path_literal_replacements=source.count(OLD))
  return compile(changed,str(path),'exec',dont_inherit=True)
 return _get_code(loader,fullname)
importlib.machinery.SourceFileLoader.get_code=relocated_code

# Refuse accidental historical TEMP reads;the shared transient lock is the sole exception.
import builtins
_open_builtin=builtins.open;_open_io=io.open
_TEMP_PREFIX=str(pathlib.Path.home()/'Documents/objective_power_grid_temp')
def _check_path(path):
 if isinstance(path,(str,bytes,os.PathLike)):
  value=os.fsdecode(path)
  if value.startswith(_TEMP_PREFIX+'/') and value!=_TEMP_PREFIX+'/.gpu0.lock':raise RuntimeError('Unrelocated TEMP dependency:'+value)
def guarded_open(path,*args,**kwargs):_check_path(path);return _open_builtin(path,*args,**kwargs)
def guarded_io_open(path,*args,**kwargs):_check_path(path);return _open_io(path,*args,**kwargs)
builtins.open=guarded_open;io.open=guarded_io_open

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
n=load_module('hec_v3_feature_native_v2',ARCHIVE/'development/hec_git_sync/development/moe_sboed/hec_v2_router_only_20261112/train_router.py')
# The frozen actor implementation imports v3_native; provide the verified relocated interface.
import types
bridge=types.ModuleType('v3_native');bridge.n=n;bridge.ROOT=ARCHIVE;bridge.WORK=ARCHIVE/'development/hec_git_sync';sys.modules['v3_native']=bridge
sys.path.insert(0,str(ARCHIVE/'development/hec_git_sync/development/moe_sboed/hec_canonical_evaluation_20261114'))
from protocol import load as canonical_load,LOCK
from resolution import signal,paired
import numpy as np
import torch
from hec_policy import COST

def setup():n.original.setup()
@contextlib.contextmanager
def lease():
 files=[]
 try:
  for p in [MAIN/'.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',pathlib.Path.home()/'Documents/objective_power_grid_temp/.gpu0.lock']:
   p.parent.mkdir(exist_ok=True,parents=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);files.append(f)
  import subprocess
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if active:raise BlockingIOError(active)
  yield
 finally:
  for f in files:f.close()

def old_results():return ARCHIVE/'scratch_runs/hec_v3_fulltrain_20261114_run01'
def parent():return n.load_parent()
def snapshot_sources():return {str(p.relative_to(MAIN)):sha(p) for p in HERE.glob('*') if p.is_file()}
def relocation_receipt(path):put(path,_RELOCATED)
def verify_references():
 cert=json.loads((ARCHIVE/'development/hec_git_sync/reports/hec_canonical_evaluation_20261114/VERIFICATION.json').read_text())
 assert cert['protocol']==LOCK and all(v['max_abs_error']==0 for v in cert['historical_reproduction'].values())
 assert sha(n.BASE/'arms/candidate/actor.pt')==n.ACTOR_SHA and sha(n.BASE/'arms/candidate/runtime.pkl')==n.RUNTIME_SHA
 for name in ['V2','V2_uniform','dad','vsoed','rl_sboed','scorebed']:
  src=n.BASE/'evaluation'/('candidate_scores.npy' if name=='V2' else name+'_scores.npy')
  if name=='V2_uniform':src=ARCHIVE/'scratch_runs/moe_hec_v2_internal_ablation/uniform_scores.npy'
  assert sha(src)==cert['historical_reproduction'][name]['saved_scores_sha256']
 canonical_load()
 return cert
