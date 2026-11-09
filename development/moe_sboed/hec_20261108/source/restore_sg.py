"""Restore verified successful SG full runtime, including replay/Adam/RNG."""
import sys,io
from native_common import *
SG=R/'moe_specialization_guided_eig_20261108';COL=R/'moe_collaborative_soft_queue_20261108'
sys.path.append(str(SG/'source'));sys.path.append(str(COL/'source'))
import train_soft_pair,train_sg_pair
CFG=json.loads((O/'CONFIG_LOCKED.json').read_text())
def restore(device='cuda'):
 p=SG/'arms/candidate/runtime.pkl';assert sha(p)==CFG['parent_runtime_sha256'];assert sha(SG/'arms/candidate/actor.pt')==CFG['parent_actor_sha256'];old=torch.storage._load_from_bytes;torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=('cpu' if device=='cpu' else lambda storage,location:storage.cuda(0) if location.startswith('cuda') else storage),weights_only=False)
 try:
  with p.open('rb') as f:state=cloudpickle.load(f)
 finally:torch.storage._load_from_bytes=old
 a=state['algorithm'];assert a.update==12001 and a.discount==1 and not a.use_PCE and len(a._runtime_buffer['params'])==1000000
 assert digest(a.actor_net.state_dict())==digest(torch.load(SG/'arms/candidate/actor.pt',map_location=device,weights_only=False)['actor_state'])
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 return state
