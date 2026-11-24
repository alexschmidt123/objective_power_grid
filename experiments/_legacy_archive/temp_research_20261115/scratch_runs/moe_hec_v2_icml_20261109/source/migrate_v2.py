import copy,torch
from native_common import O,digest
from v2_policy import V2Actor
CFG=__import__('json').loads((O/'CONFIG_LOCKED.json').read_text())
def phase(model,which):
 for p in model.parameters():p.requires_grad_(True)
 model.readouts.requires_grad_(False)
 if hasattr(model,'response_readouts'):model.response_readouts.requires_grad_(False)
 if which=='B':model.history.requires_grad_(False)
 else:assert which=='C'
def migrate(a,mode,prepared=True):
 old=a.actor_net;target=a.actor_target_net
 if mode!='continuation':
  oldkeys=set(old.state_dict());before=digest(old.state_dict());model=V2Actor(old,mode=='stage',CFG['new_adapter_seed']);tgt=V2Actor(target,mode=='stage',CFG['new_adapter_seed'])
  if prepared:
   cp=torch.load(O/'preparation/prepared.pt',map_location='cuda',weights_only=False)['actor_state'];model.load_state_dict(cp);new={k:v for k,v in cp.items() if k not in oldkeys};tgt.load_state_dict(new,strict=False)
   with torch.no_grad():
    for k in [0,2]:tgt.response_fusion.weight[:,k*64:(k+1)*64].copy_(tgt.fusion[0].weight[:,128+k*64:128+(k+1)*64])
  assert digest({k:v for k,v in model.state_dict().items() if k in oldkeys})==before,'Parent parameters changed in preparation';a.actor_net=model;a.actor_target_net=tgt;a.actor_optimizer.param_groups[0]['params']=list(model.parameters())
 phase(a.actor_net,'B');return dict(parent_compatible_parameters_and_Adam_preserved=True,new_response_and_stage_parameter_Adam_fresh=mode!='continuation',mode=mode)
