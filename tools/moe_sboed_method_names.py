"""Canonical paper names;historical versions and storage paths retain provenance."""
METHODS={
 'MoE-sBOED':dict(role='proposed',training='learned-routing',aliases=['learned','full_v3','MoE-sBOED','V3 learned']),
 'MoE-sBOED-uniform':dict(role='internal-baseline',training='independently-trained-fixed-uniform',aliases=['uniform','trained_uniform','V3 uniform']),
 **{f'MoE-sBOED-E{k}':dict(role='internal-inherited-checkpoint-baseline',training='inherited-from-corresponding-learned-checkpoint;no-independent-training-claim',aliases=[f'expert{k}',f'expert_{k}',f'E{k}']) for k in range(1,5)},
 'vsOED':dict(role='external-baseline',aliases=['vsOED']),
 'RL-sBOED':dict(role='external-baseline',aliases=['rl_sboed','RL-sBOED']),
 'ScoreBED':dict(role='external-baseline',aliases=['ScoreBED']),
 'DAD':dict(role='external-baseline',aliases=['DAD'])}
ALIASES={name:name for name in METHODS}
ALIASES.update({name.lower():name for name in METHODS})
for name,info in METHODS.items():
 for alias in info.get('aliases',[]):
  if alias in ALIASES and ALIASES[alias]!=name:raise ValueError('Ambiguous alias:'+alias)
  ALIASES[alias]=name

def canonical(name):
 if name not in ALIASES:raise ValueError('Unregistered method;keep historical version as provenance:'+name)
 return ALIASES[name]

def validate():
 assert [canonical(v) for v in ['learned','uniform','expert1','expert2','expert3','expert4']]==['MoE-sBOED','MoE-sBOED-uniform',*[f'MoE-sBOED-E{k}' for k in range(1,5)]]
 assert 'V2' not in ALIASES and 'V3' not in ALIASES
 assert len(METHODS)==10 and len([v for v in METHODS.values() if v['role']=='internal-inherited-checkpoint-baseline'])==4
 return dict(status='PASS',independent_training_families=['MoE-sBOED','MoE-sBOED-uniform','vsOED','RL-sBOED','ScoreBED','DAD'],inherited_internal_baselines=[f'MoE-sBOED-E{k}' for k in range(1,5)])
if __name__=='__main__':
 import json;print(json.dumps(validate(),indent=2))
