"""Canonical paper names;historical versions and storage paths retain provenance."""
METHODS={
 'moe-sboed':dict(role='proposed',training='learned-routing',aliases=['learned','full_v3','MoE-sBOED','V3 learned']),
 'moe-sboed-uniform':dict(role='internal-baseline',training='independently-trained-fixed-uniform',aliases=['uniform','trained_uniform','V3 uniform']),
 **{f'moe-sboed-e{k}':dict(role='internal-inherited-checkpoint-baseline',training='inherited-from-corresponding-learned-checkpoint;no-independent-training-claim',aliases=[f'expert{k}',f'expert_{k}',f'E{k}']) for k in range(1,5)},
 'vsoed':dict(role='external-baseline',aliases=['vsOED']),
 'rl-sboed':dict(role='external-baseline',aliases=['rl_sboed','RL-sBOED']),
 'scorebed':dict(role='external-baseline',aliases=['ScoreBED']),
 'dad':dict(role='external-baseline',aliases=['DAD'])}
ALIASES={name:name for name in METHODS}
for name,info in METHODS.items():
 for alias in info.get('aliases',[]):
  if alias in ALIASES and ALIASES[alias]!=name:raise ValueError('Ambiguous alias:'+alias)
  ALIASES[alias]=name

def canonical(name):
 if name not in ALIASES:raise ValueError('Unregistered method;keep historical version as provenance:'+name)
 return ALIASES[name]

def validate():
 assert [canonical(v) for v in ['learned','uniform','expert1','expert2','expert3','expert4']]==['moe-sboed','moe-sboed-uniform',*[f'moe-sboed-e{k}' for k in range(1,5)]]
 assert 'V2' not in ALIASES and 'V3' not in ALIASES
 assert len(METHODS)==10 and len([v for v in METHODS.values() if v['role']=='internal-inherited-checkpoint-baseline'])==4
 return dict(status='PASS',independent_training_families=['moe-sboed','moe-sboed-uniform','vsoed','rl-sboed','scorebed','dad'],inherited_internal_baselines=[f'moe-sboed-e{k}' for k in range(1,5)])
if __name__=='__main__':
 import json;print(json.dumps(validate(),indent=2))
