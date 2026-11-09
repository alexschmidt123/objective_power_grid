import os,sys,pathlib,hashlib,json,io,math,time
sys.dont_write_bytecode=True
import torch
O=pathlib.Path(__file__).resolve().parents[1];R=pathlib.Path.home()/'Documents/objective_power_grid_temp/scratch_runs';P=R/'moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101';sys.path.insert(0,str(P/'source'));sys.path.insert(0,str(R/'moe_conditional_strategy_audit_20261105/dependencies'))
import cloudpickle
import run_actor as original
from moe_actor import digest
PARENT=P/'checkpoints/update10001_recovered_final_runtime.pkl';PARENT_SHA='f6b60505215273589d833dae05b9254fc6781bee52641ebbc885a16966638a8e'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return h.hexdigest()
def put(p,x):
 def cv(v):
  if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
  if hasattr(v,'item'):return v.item()
  raise TypeError(type(v).__name__)
 pathlib.Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=cv))
def load(device='cpu'):
 assert sha(PARENT)==PARENT_SHA
 old=torch.storage._load_from_bytes;torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=('cpu' if device=='cpu' else lambda storage,location:storage.cuda(0) if location.startswith('cuda') else storage),weights_only=False)
 try:
  with PARENT.open('rb') as f:state=cloudpickle.load(f)
 finally:torch.storage._load_from_bytes=old
 a=state['algorithm'];assert a.update==10001 and not a.use_PCE and a.discount==.9 and a.n_batch==10000 and a.n_newtraj==1000;assert abs(a.actor_optimizer.param_groups[0]['lr']-.001*.9999**10001)<1e-12
 assert digest(a.actor_net.state_dict())==json.loads((O/'evidence/INPUT_PROVENANCE.json').read_text())['actor_parameter_digest']
 assert all(float(s['step'])==10001 for s in a.actor_optimizer.state.values())
 for opt in allopts(a):opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
 return state
def allopts(a):return [a.actor_optimizer,a.critic_optimizer]+[o for oo in a.post_approx.poi_post_optimizers.values() for o in oo]
def norm(grads):return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))
