import os,sys,json,time,math,hashlib,pathlib,io
sys.dont_write_bytecode=True
import numpy as np,torch
R=pathlib.Path.home()/'Documents/objective_power_grid_temp/scratch_runs';OLD=R/'moe_hec_v2_icml_20261109';O=pathlib.Path(os.environ.get('MOE_V3_RUN_ROOT',str(R/'moe_sboed_v3_model_informed_20261110')));CFG=json.loads((O/'CONFIG_LOCKED.json').read_text());os.environ['MOE_V2_RUN_ROOT']=str(OLD);sys.path.append(str(OLD/'source'))
import restore_v1,train_v2
from native_common import original,P,sha,put,digest,allopts,norm,cloudpickle
from hec_policy import COST,mean_signal,jacobian
from sg_policy import frozen_buffers
from soft_policy import prefix_inputs
from normalized_reward import normalize_native_post

def restore(device='cuda'):
 runtime=OLD/'arms/candidate/runtime.pkl';cp=OLD/'arms/candidate/actor.pt';receipt=json.loads((OLD/'arms/candidate/RESULT.json').read_text());assert sha(runtime)==receipt['runtime_sha256'] and sha(cp)==receipt['actor_sha256']
 old=torch.storage._load_from_bytes;torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=('cpu' if device=='cpu' else lambda storage,loc:storage.cuda(0) if loc.startswith('cuda') else storage),weights_only=False)
 try:
  with runtime.open('rb') as f:s=cloudpickle.load(f)
 finally:torch.storage._load_from_bytes=old
 a=s['algorithm'];assert a.update==16001 and a.discount==1 and not a.use_PCE and a.n_batch==10000 and len(a._runtime_buffer['params'])==1000000;assert digest(a.actor_net.state_dict())==digest(torch.load(cp,map_location=device,weights_only=False)['actor_state'])
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 return s

def stats(x):
 from parent_evaluation import stats as f
 return f(np.asarray(x))
def holm(tests):
 from parent_evaluation import holm as f
 f(tests,'Holm_p');return tests
