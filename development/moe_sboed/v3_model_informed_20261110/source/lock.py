"""Verify provenance and seal the authorized configuration before any update."""
from common import *
assert not any((O/p).exists() for p in ['PRETRAIN_MANIFEST.json','preparation','arms','evaluation','evidence/TRAINING_STARTED.json','evidence/LAUNCH_RECEIPT.json'])
assert CFG['status']=='VALIDATED_FIXED_PROTOCOL' and CFG['updates_per_arm']==2000 and CFG['arms']==['stage','candidate'] and not CFG['new_rewards'] and CFG['main_auxiliary_weight']==0
for n in ['IMPLEMENTATION_CHECKS.json','CONDITION_PREFLIGHT.json','EVALUATION_CHECKS.json','CONDITIONAL_CHECKS.json','dad_preflight.json','rl_sboed_preflight.json','scorebed_preflight.json']:
 assert json.loads((O/'evidence'/n).read_text())['status']=='PASS',n
assert json.loads((O/'evidence/GRADIENT_FIXTURES.json').read_text())['status']=='PASS_COMPONENT_FIXTURES_ONLY'
sync=json.loads((O/'evidence/GITHUB_SYNC_RECEIPT.json').read_text());assert sync['status']=='VERIFIED' and sync['labpc']['commit']==sync['ubuntu']['commit']==sync['github_commit']
refs=json.loads((OLD/'evidence/REFERENCE_SOURCE_INPUTS.json').read_text());sealed=json.loads((OLD/'ARTIFACT_SHA256.json').read_text())['files']
for rel,h in sealed.items():
 if rel.startswith(('source/','inputs/')) or rel in ['arms/candidate/actor.pt','arms/candidate/runtime.pkl','arms/candidate/RESULT.json','CONFIG_LOCKED.json','EIG_COMPARISON.md','NEXT_RESEARCH_DECISION.md']:refs[str(OLD/rel)]=h
refs[str(OLD/'ARTIFACT_SHA256.json')]=sha(OLD/'ARTIFACT_SHA256.json')
prior=R/'moe_sboed_v3_competitive_20261110';seal=json.loads((prior/'PREFLIGHT_MANIFEST.json').read_text())
for rel,h in seal['files'].items():refs[str(prior/rel)]=h
refs[str(prior/'PREFLIGHT_MANIFEST.json')]=sha(prior/'PREFLIGHT_MANIFEST.json')
for path,h in refs.items():assert sha(path)==h,path
put(O/'evidence/PROTECTED_INPUTS_BEFORE.json',refs);files={str(p.relative_to(O)):sha(p) for folder in ['source','evidence','inputs'] for p in (O/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for name in ['CONFIG_LOCKED.json','PROTOCOL_LOCK.md','V3_ALGORITHM_AND_NOVELTY.md','EXPERT_SPECIALIZATION_DEFINITION.md','ROUTER_RESPONSIBILITY_RULE.md','POLICY_GRADIENT_DERIVATION.md']:files[name]=sha(O/name)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_PREPARATION_AND_TRAINING',time=time.time(),git_commit=sync['github_commit'],files=files,protected_files=len(refs),scientific_updates=0,preparation_updates=0));print('V3_PROTOCOL_LOCKED',sync['github_commit'],len(files),len(refs))
