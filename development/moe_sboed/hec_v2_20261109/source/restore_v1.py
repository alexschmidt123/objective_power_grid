import sys,io
from native_common import *
V1=R/'moe_hec_sboed_20261108';SG=R/'moe_specialization_guided_eig_20261108';COL=R/'moe_collaborative_soft_queue_20261108'
sys.path.append(str(SG/'source'));sys.path.append(str(COL/'source'))
import train_soft_pair,train_sg_pair,train_pair
CFG=json.loads((O/'CONFIG_LOCKED.json').read_text())
def restore(device='cuda'):
 runtime=V1/'arms/candidate/runtime.pkl';assert sha(runtime)==CFG['parent_runtime_sha256'] and sha(V1/'arms/candidate/actor.pt')==CFG['parent_actor_sha256'];old=torch.storage._load_from_bytes;torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=('cpu' if device=='cpu' else lambda storage,loc:storage.cuda(0) if loc.startswith('cuda') else storage),weights_only=False)
 try:
  with runtime.open('rb') as f:state=cloudpickle.load(f)
 finally:torch.storage._load_from_bytes=old
 a=state['algorithm'];assert a.update==14001 and a.discount==1 and not a.use_PCE and len(a._runtime_buffer['params'])==1000000 and a.actor_net.dynamic
 assert digest(a.actor_net.state_dict())==digest(torch.load(V1/'arms/candidate/actor.pt',map_location=device,weights_only=False)['actor_state'])
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 return state
