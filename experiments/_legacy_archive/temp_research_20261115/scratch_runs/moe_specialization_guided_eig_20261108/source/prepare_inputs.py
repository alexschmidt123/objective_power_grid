"""Frozen same-benchmark references and baseline provenance inventory."""
import csv,shutil,json
from native_common import *
from restore_parent import PARENT_ROOT,ACTOR_SHA,RUNTIME_SHA
O.joinpath('inputs').mkdir(exist_ok=True);assert sha(PARENT_ROOT/'arms/candidate/actor.pt')==ACTOR_SHA;assert sha(PARENT_ROOT/'arms/candidate/runtime.pkl')==RUNTIME_SHA
copies={}
for name,source in {'parent':PARENT_ROOT/'arms/candidate/actor.pt','historical_control':PARENT_ROOT/'arms/control/actor.pt','reference':PARENT_ROOT/'inputs/reference.pt','original':PARENT_ROOT/'inputs/original.pt','vsoed':PARENT_ROOT/'inputs/vsoed.pt','ordinary':PARENT_ROOT/'inputs/ordinary.pt'}.items():
 dst=O/f'inputs/{name}.pt';assert not dst.exists();shutil.copyfile(source,dst);assert sha(source)==sha(dst);copies[name]=dict(source=str(source),sha256=sha(source))
for name in ['reference_native_posterior.pt']:
 src=PARENT_ROOT/'inputs'/name;shutil.copyfile(src,O/'inputs'/name)
B=R/'journal_LF_T30_native_final_20261102T052000Z';rows=[r for r in csv.DictReader((B/'production_manifest.csv').open()) if r['seed']=='101' and r['method'] in ['dad','rl_sboed','scorebed']]
for row in rows:
 assert sha(row['checkpoint'])==row['checkpoint_sha256'];row['frozen_source_root']=str(B/'source');row['historical_not_compute_matched']=True
 if row['method']=='scorebed':
  arrays=pathlib.Path(row['checkpoint']).with_name('scorebed_official_policy.npz');row['arrays_sha256']=sha(arrays);row['arrays_path']=str(arrays)
put(O/'inputs/HISTORICAL_BASELINES.json',dict(rows=rows,physics=json.loads((B/'physical.json').read_text()),original_protocol=json.loads((B/'FORMAL_PROTOCOL.json').read_text()),source_root=str(B/'source'),scorer_sha256=sha(B/'source/src/research/location_evaluation.py')));put(O/'evidence/INPUT_PROVENANCE.json',dict(status='VERIFIED',parent_actor_sha256=ACTOR_SHA,parent_runtime_sha256=RUNTIME_SHA,copies=copies,baseline_inventory=rows,training_updates=0));print('SG_INPUT_PROVENANCE_VERIFIED')
