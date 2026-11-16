"""Seal source, settings, successful validation and preserved inputs."""
from common import *
assert not (O/'PRETRAIN_MANIFEST.json').exists()
assert not any((O/name).exists() for name in ['arms','preparation','evaluation','evidence/TRAINING_STARTED.json'])
for name in ['IMPLEMENTATION_CHECKS','NATIVE_FULL_BATCH_CHECKS','PREPARATION_CHECKS','EVALUATION_CHECKS','dad_preflight','rl_sboed_preflight','scorebed_preflight']:
    assert json.loads((O/f'evidence/{name}.json').read_text())['status']=='PASS',name
sync=json.loads((O/'evidence/GITHUB_SYNC_RECEIPT.json').read_text())
assert sync['status']=='VERIFIED' and sync['github_commit']==sync['labpc']['commit']==sync['ubuntu']['commit']
refs=protected();refs[str(R/'cer_moe_sboed_v3_20261111_local_values/evidence/signatures.npz')]=sha(R/'cer_moe_sboed_v3_20261111_local_values/evidence/signatures.npz')
inventory=json.loads((O/'inputs/HISTORICAL_BASELINES.json').read_text())
for row in inventory['rows']:
    for key in ['checkpoint','arrays_path']:
        if row.get(key):refs[row[key]]=sha(row[key])
put(O/'evidence/PROTECTED_INPUTS_BEFORE.json',refs)
files={str(p.relative_to(O)):sha(p) for folder in ['source','inputs','evidence'] for p in (O/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for name in ['CONFIG_LOCKED.json','METHOD_AND_PROTOCOL.md']:files[name]=sha(O/name)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_TRAINING',git_commit=sync['github_commit'],time=time.time(),files=files,
    exactly_one_candidate_and_control=True,scientific_optimizer_updates=0,protected_files=len(refs)))
print('CER_REVISED_PROTOCOL_LOCKED',sync['github_commit'])
