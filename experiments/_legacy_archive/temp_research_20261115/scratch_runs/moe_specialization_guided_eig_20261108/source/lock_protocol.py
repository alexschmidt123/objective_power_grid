"""Seal science before the only authorized paired continuation."""
from native_common import *
from restore_parent import PARENT_ROOT,RUNTIME_SHA
assert not (O/'PRETRAIN_MANIFEST.json').exists()
for n in ['SG_CHECKS.json','NATIVE_LOOP_CHECK.json','EVALUATION_TEST_RESULTS.json','EVALUATION_PATH_CHECKS.json','dad_preflight.json','rl_sboed_preflight.json','scorebed_preflight.json']:assert json.loads((O/'evidence'/n).read_text())['status']=='PASS',n
before=json.loads((PARENT_ROOT/'evidence/ORIGINALS_BEFORE.json').read_text())
for root,files in before.items():
 p=pathlib.Path(root)
 assert {str(v.relative_to(p)):sha(v) for v in p.rglob('*') if v.is_file()}==files,root
for root in [PARENT_ROOT,R/'moe_task_competence_recovery_20261108']:
 assert root.exists();before[str(root)]={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
put(O/'evidence/ORIGINALS_BEFORE.json',before);refs={str(p):sha(p) for p in (P/'source').rglob('*') if p.is_file()};inventory=json.loads((O/'inputs/HISTORICAL_BASELINES.json').read_text())
for p in pathlib.Path(inventory['source_root']).rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts:refs[str(p)]=sha(p)
for row in inventory['rows']:
 refs[row['checkpoint']]=row['checkpoint_sha256']
 if row.get('arrays_path'):refs[row['arrays_path']]=row['arrays_sha256']
refs[str(PARENT_ROOT/'arms/candidate/runtime.pkl')]=RUNTIME_SHA
put(O/'evidence/REFERENCE_SOURCE_INPUTS.json',refs)
files={}
for directory in ['source','inputs','evidence']:
 for p in (O/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(O))]=sha(p)
for name in ['CONFIG_LOCKED.json','PROTOCOL_LOCK.md','NEW_ALGORITHM_DERIVATION.md','ARCHITECTURE.md','SPECIALIZATION_TRAINING.md','EIG_ROUTING_AND_FUSION.md','TASK_SPECIFICATION.md']:files[name]=sha(O/name)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_TRAINING',time=time.time(),files=files,scientific_optimizer_updates=0,historical_roots=len(before),historical_files=sum(map(len,before.values())),one_fixed_pair=True));print('SG_PROTOCOL_LOCKED')
