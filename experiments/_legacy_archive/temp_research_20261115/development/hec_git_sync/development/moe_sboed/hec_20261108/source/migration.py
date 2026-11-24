import copy,torch
from hec_policy import HECActor
from native_common import O,digest
from prepare_specialists import create_model

def migrate(a,dynamic,prepared=True):
 old=a.actor_net;old_target=a.actor_target_net;history=old.encoder;target_history=old_target.encoder;model=create_model(a)
 if prepared:model.load_state_dict(torch.load(O/'preparation/prepared.pt',map_location='cuda',weights_only=False)['actor_state'])
 model.history=history;target=copy.deepcopy(model);target.history=target_history;model.dynamic=dynamic;target.dynamic=dynamic;a.actor_net=model;a.actor_target_net=target;params=list(model.parameters());keep=set(params)
 for p in list(a.actor_optimizer.state):
  if p not in keep:del a.actor_optimizer.state[p]
 a.actor_optimizer.param_groups[0]['params']=params;model.set_phase('B');a._soft_tasks.enabled=False;a.discount=1.;return dict(history_parameters_and_Adam_preserved=True,removed_incompatible_old_expert_and_fusion_Adam=True,new_experts_prepared_shared_weights=True,new_parameter_Adam_fresh=True,frozen_SG_belief_critic_are_buffers=True)
