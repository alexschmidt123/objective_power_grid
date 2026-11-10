from native_common import *
from restore_v1 import V1,CFG
assert not (O/'PRETRAIN_MANIFEST.json').exists()
for name in ['V2_CHECKS.json','REWARD_PREFIX_CHECKS.json','EVALUATION_CHECKS.json','dad_preflight.json','rl_sboed_preflight.json','scorebed_preflight.json']:assert json.loads((O/'evidence'/name).read_text())['status']=='PASS',name
sync=json.loads((O/'evidence/GITHUB_SYNC_RECEIPT.json').read_text());assert sync['status']=='VERIFIED' and sync['labpc']['commit']==sync['ubuntu']['commit']==sync['github_commit'];roots=json.loads((V1/'evidence/ORIGINALS_BEFORE.json').read_text())
for root,before in roots.items():
 p=pathlib.Path(root);now={str(v.relative_to(p)):sha(v) for v in p.rglob('*') if v.is_file() and '__pycache__' not in v.parts};expected={k:v for k,v in before.items() if '__pycache__' not in pathlib.Path(k).parts};assert now==expected,root
for p in [V1,R/'moe_hec_runtime_efficiency_audit_20261109']:roots[str(p)]={str(v.relative_to(p)):sha(v) for v in p.rglob('*') if v.is_file() and '__pycache__' not in v.parts}
put(O/'evidence/ORIGINALS_BEFORE.json',roots);refs=json.loads((V1/'evidence/REFERENCE_SOURCE_INPUTS.json').read_text());refs[str(V1/'arms/candidate/runtime.pkl')]=CFG['parent_runtime_sha256']
for p,h in refs.items():assert sha(p)==h,p
put(O/'evidence/REFERENCE_SOURCE_INPUTS.json',refs);files={}
for folder in ['source','inputs','evidence']:
 for p in (O/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(O))]=sha(p)
for name in ['CONFIG_LOCKED.json','PROTOCOL_LOCK.md','SPECIALIST_DESIGN.md','COMMANDER_DESIGN.md','ALGORITHM_DERIVATION.md']:files[name]=sha(O/name)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_ANY_PREPARATION_OR_TRAINING',time=time.time(),git_commit=sync['github_commit'],files=files,three_fixed_arms=True,historical_roots=len(roots),scientific_optimizer_updates=0));print('V2_PROTOCOL_LOCKED')
