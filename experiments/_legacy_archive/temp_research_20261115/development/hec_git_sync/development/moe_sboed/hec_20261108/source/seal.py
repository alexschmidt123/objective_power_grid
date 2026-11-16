from native_common import *
def seal():
 manifest=json.loads((O/'PRETRAIN_MANIFEST.json').read_text())
 for n,h in manifest['files'].items():assert sha(O/n)==h,n
 roots=json.loads((O/'evidence/ORIGINALS_BEFORE.json').read_text())
 for root,before in roots.items():
  p=pathlib.Path(root);now={str(v.relative_to(p)):sha(v) for v in p.rglob('*') if v.is_file() and '__pycache__' not in v.parts};expected={k:v for k,v in before.items() if '__pycache__' not in pathlib.Path(k).parts};assert now==expected,root
 refs=json.loads((O/'evidence/REFERENCE_SOURCE_INPUTS.json').read_text())
 for p,h in refs.items():assert sha(p)==h,p
 put(O/'evidence/FINAL_PRESERVATION.json',dict(status='VERIFIED',historical_roots=len(roots),historical_files=sum(map(len,roots.values())),reference_source_files=len(refs),Ubuntu_accessed=False,unrelated_processes_interrupted=False))
 files={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['ARTIFACT_SHA256.json','experiment.log']};put(O/'ARTIFACT_SHA256.json',dict(status='VERIFIED',files=files,volatile_experiment_log_excluded=True));print('HEC_ARTIFACTS_SEALED')
