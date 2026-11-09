"""Verify protected artifacts and seal HEC science before preparation."""
from native_common import *
from restore_sg import SG,CFG
assert not (O/'PRETRAIN_MANIFEST.json').exists()
for n in ['CHECKS.json','EVALUATION_CHECKS.json','dad_preflight.json','rl_sboed_preflight.json','scorebed_preflight.json']:assert json.loads((O/'evidence'/n).read_text())['status']=='PASS',n
before=json.loads((SG/'evidence/ORIGINALS_BEFORE.json').read_text())
for root,files in before.items():
 p=pathlib.Path(root);actual={str(v.relative_to(p)):sha(v) for v in p.rglob('*') if v.is_file() and '__pycache__' not in v.parts};expected={k:v for k,v in files.items() if '__pycache__' not in pathlib.Path(k).parts};assert actual==expected,root
assert json.loads((SG/'evidence/FINAL_PRESERVATION.json').read_text())['status']=='VERIFIED'
before[str(SG)]={str(p.relative_to(SG)):sha(p) for p in SG.rglob('*') if p.is_file() and '__pycache__' not in p.parts};put(O/'evidence/ORIGINALS_BEFORE.json',before);refs=json.loads((SG/'evidence/REFERENCE_SOURCE_INPUTS.json').read_text())
for p,h in refs.items():assert sha(p)==h,p
refs[str(SG/'arms/candidate/runtime.pkl')]=CFG['parent_runtime_sha256'];put(O/'evidence/REFERENCE_SOURCE_INPUTS.json',refs);files={}
for directory in ['source','inputs','evidence']:
 for p in (O/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(O))]=sha(p)
for name in ['CONFIG_LOCKED.json','PROTOCOL_LOCK.md','ARCHITECTURE.md','MATHEMATICS.md','SPECIALIST_TASKS.md','COMMANDER_RULE.md','TASK_INPUT_WITH_WRAPPER.txt']:files[name]=sha(O/name)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_PREPARATION',time=time.time(),files=files,scientific_optimizer_updates=0,historical_roots=len(before),historical_files=sum(map(len,before.values())),one_fixed_pair=True));print('HEC_PROTOCOL_LOCKED')
