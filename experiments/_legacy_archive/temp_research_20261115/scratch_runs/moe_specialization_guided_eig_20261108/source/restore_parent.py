"""Verified full parent state;no fresh policy or optimizer replacement."""
import sys,io,pathlib,json,copy
sys.dont_write_bytecode=True
from native_common import *
PARENT_ROOT=R/'moe_collaborative_soft_queue_20261108';sys.path.append(str(PARENT_ROOT/'source'))
import train_soft_pair
ACTOR_SHA='837bcca20af6521e2db0a589d48695efd9b6620407ffc85729a891a5d81986b3';RUNTIME_SHA='77db8045f1d1c5beae009c465ecdaee5ab74ff1ebcc41e4ff3c6018eb1642981'
def restore(device='cuda'):
 runtime=PARENT_ROOT/'arms/candidate/runtime.pkl';assert sha(runtime)==RUNTIME_SHA and sha(PARENT_ROOT/'arms/candidate/actor.pt')==ACTOR_SHA;old=torch.storage._load_from_bytes
 torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=('cpu' if device=='cpu' else lambda storage,location:storage.cuda(0) if location.startswith('cuda') else storage),weights_only=False)
 try:
  with runtime.open('rb') as f:state=cloudpickle.load(f)
 finally:torch.storage._load_from_bytes=old
 a=state['algorithm'];assert a.update==10001 and not a.use_PCE and a.discount==.9 and hasattr(a,'_runtime_buffer');assert len(a._runtime_buffer['params'])==1000000
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 cp=torch.load(PARENT_ROOT/'arms/candidate/actor.pt',map_location=device,weights_only=False);assert digest(a.actor_net.state_dict())==digest(cp['actor_state']);assert all(float(s['step'])==10001 for s in a.actor_optimizer.state.values());return state
