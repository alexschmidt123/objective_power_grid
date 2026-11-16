"""Reuse audited tensor references;do not deserialize Python-version-specific parent."""
import shutil
from native_common import *
old=R/'moe_information_specialized_option_20261107';m=json.loads((old/'PRETRAIN_MANIFEST.json').read_text());assert json.loads((old/'evidence/FINAL_PRESERVATION.json').read_text())['status']=='VERIFIED';copied={}
for p in (old/'inputs').iterdir():
 if p.is_file():
  assert sha(p)==m['files']['inputs/'+p.name];dst=O/'inputs'/p.name;assert not dst.exists(),dst;shutil.copyfile(p,dst);assert sha(dst)==sha(p);copied[str(dst)]=sha(dst)
for name in ['INPUT_PROVENANCE.json','REFERENCE_VERIFICATION.json']:shutil.copyfile(old/'evidence'/name,O/'evidence'/name)
assert sha(PARENT)==PARENT_SHA
for name,h in json.loads((O/'evidence/INPUT_PROVENANCE.json').read_text())['source_hashes'].items():assert sha(P/name)==h
put(O/'evidence/REFERENCE_REUSE.json',dict(status='VERIFIED',source_manifest_sha256=sha(old/'PRETRAIN_MANIFEST.json'),source_root=str(old),copied=copied,parent_runtime_deserialized=False,reason='historical runtime cloudpickle Python-version incompatibility;reuse previously audited tensorcopies',trained_Adam_or_replay_used=False,training_updates=0));print('AUDITED_TENSOR_REFERENCES_VERIFIED')
