from common import *
assert not (O/'PRETRAIN_MANIFEST.json').exists()
assert not any((O/n).exists() for n in ['arms','preparation','evaluation','evidence/TRAINING_STARTED.json'])
for name in ['IMPLEMENTATION_CHECKS','PHYSICS_CHECKS','PREPARATION_CHECKS','EVALUATION_CHECKS','dad_preflight','rl_sboed_preflight','scorebed_preflight']:
    assert json.loads((O/f'evidence/{name}.json').read_text())['status']=='PASS',name
sync=json.loads((O/'evidence/GITHUB_SYNC_RECEIPT.json').read_text())
assert sync['status']=='VERIFIED' and sync['github_commit']==sync['labpc']['commit']==sync['ubuntu']['commit']
refs=protected()
for p in (P/'source').rglob('*.py'):
    if '__pycache__' not in p.parts:refs[str(p)]=sha(p)
for p in [P/'frozen_protocol.json',P/'source_manifest.json']:
    if p.exists():refs[str(p)]=sha(p)
for p in (OLD/'inputs').rglob('*'):
    if p.is_file():refs[str(p)]=sha(p)
refs[str(OLD/'arms/candidate/RESULT.json')]=sha(OLD/'arms/candidate/RESULT.json')
for row in json.loads((O/'inputs/HISTORICAL_BASELINES.json').read_text())['rows']:
    for key in ['checkpoint','arrays_path']:
        if row.get(key):refs[row[key]]=sha(row[key])
put(O/'evidence/PROTECTED_INPUTS_BEFORE.json',refs)
files={str(p.relative_to(O)):sha(p) for folder in ['source','inputs','evidence'] for p in (O/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for name in ['CONFIG_LOCKED.json','RUN_INDEX.md','HEC_V2_BASELINE_AUDIT.md','RR_MOE_ARCHITECTURE.md','ROUTING_GRADIENT_DERIVATION.md','SPECIALIST_ADAPTATION.md','PROTOCOL_LOCK.md','IMPLEMENTATION_TESTS.md']:files[name]=sha(O/name)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_TRAINING',time=time.time(),git_commit=sync['github_commit'],files=files,scientific_updates=0,protected_files=len(refs),single_candidate_single_control=True))
print('RR_PROTOCOL_LOCKED',sync['github_commit'])
