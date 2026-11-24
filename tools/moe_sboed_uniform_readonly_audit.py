import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,math
import numpy as np
import torch
torch.set_num_threads(2)
M=Path.home()/'Documents/objective_power_grid';I=M/'experiments/location_finding/hec_v3_study_20261115';S=M/'src/research/hec_v3_study'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prefixes=['router.','prior_coordinator.','prior_response_gate.'];rows=[]
parent=torch.load(M/'experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109/arms/candidate/actor.pt',map_location='cpu',weights_only=True)['actor_state']
for seed in range(101,106):
 roots={a:I/f'seed{seed}_{a}' for a in ['learned','uniform']};initial={a:read(p/'INITIALIZATION.json') for a,p in roots.items()};assert initial['learned']['actor_digest']==initial['uniform']['actor_digest']
 states={a:torch.load(p/'joint_actor.pt',map_location='cpu',weights_only=True)['actor_state'] for a,p in roots.items()};assert states['learned'].keys()==states['uniform'].keys();assert all(states['learned'][k].shape==states['uniform'][k].shape and states['learned'][k].dtype==states['uniform'][k].dtype for k in states['learned'])
 item={'seed':seed,'paired_initial_state_exact_digest':initial['learned']['actor_digest'],'identical_state_shapes_and_dtypes':True}
 for arm,p in roots.items():
  res=read(p/'RESULTS.json');cfg=read(p/'CONFIG.json');manifest=read(p/'SOURCE_MANIFEST.json');assert all(sha(M/k)==v for k,v in manifest.items() if k.startswith("src/"))
  assert res['updates']==2000 and res['new_training_episodes']==2000000 and res['new_training_observations']==60000000
  phases=[read(p/(q+'_TRAINING.json')) for q in ['warmup','joint']];assert all(q['frozen_state_preserved'] for q in phases)
  logs=[json.loads(x) for x in (p/'TRAINING_METRICS.jsonl').read_text().splitlines()];assert all(all(math.isfinite(v) for v in l['gradient_group_norms'].values()) for l in logs)
  groups=['experts.0','experts.1','experts.2','experts.3','decoder.weight','decoder.response_weight','decoder.action_layer','context.commander']
  nonzero={k:sum(l['gradient_group_norms'].get(k,0)>0 for l in logs) for k in groups};assert all(nonzero.values())
  assert all(l['gradient_group_norms'].get('context.history',0)==0 for l in logs if l['phase']=='warmup')
  assert any(l['gradient_group_norms'].get('context.history',0)>0 for l in logs if l['phase']=='joint')
  cp=states[arm];warm=torch.load(p/'warmup_actor.pt',map_location='cpu',weights_only=True)['actor_state']
  ev=next(iter(res['evaluations'].values()));g=np.load(p/(arm+'_routing.npz'))['gates'];assert g.shape==(2000,30,4)
  detail=dict(trainable_warmup=phases[0]['trainable_parameters'],trainable_joint=phases[1]['trainable_parameters'],evaluation_protocol=ev['protocol'],logged_nonzero_gradient_counts=nonzero,history_only_joint=True,source_manifest_matches_current=True,training_seconds=res['training_seconds'],updates=res['updates'],counts={k:sum(q['counts'][k] for q in phases) for k in ['actor','critic','posterior']})
  if arm=='uniform':
   assert np.all(g==.25)
   assert not any(any(k.startswith(q[:-1]) for q in prefixes) for l in logs for k in l['gradient_group_norms'])
   checks={}
   for pref in prefixes:
    keys=[k for k in cp if k.startswith(pref)];assert keys
    assert all(torch.equal(cp[k],warm[k]) for k in keys)
    if pref=='router.':
     assert torch.count_nonzero(cp['router.2.weight'])==0 and torch.count_nonzero(cp['router.2.bias'])==0
     torch.manual_seed(202611150000+seed)
     for width in [13,28,20,26]:
      tmp=torch.nn.Sequential(torch.nn.Linear(width,64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,128))
     expected=torch.nn.Sequential(torch.nn.Linear(147,64),torch.nn.SiLU(),torch.nn.Linear(64,4))
     assert torch.equal(cp['router.0.weight'],expected[0].weight) and torch.equal(cp['router.0.bias'],expected[0].bias)
     # First-layer random weights are protected by per-phase frozen digest, not replayed RNG here.
    else:
     old='coordinator.' if pref=='prior_coordinator.' else 'response_gate.'
     assert all(torch.equal(cp[k],parent[old+k[len(pref):]]) for k in keys)
    checks[pref]={'warmup_to_final_bitwise_unchanged':True,'tensor_elements':sum(cp[k].numel() for k in keys),'initial_state_verified':True,'initial_provenance':'inherited parent' if pref!='router.' else 'exact seeded first layer and zero final layer'}
   detail.update(evaluation_exact_quarter_entries=int(g.size),routing_frozen_checks=checks,router_absent_from_logged_gradients=True)
  changed={pref:sum(not torch.equal(cp[k],warm[k]) for k in cp if k.startswith(pref)) for pref in ['experts.0.','experts.1.','experts.2.','experts.3.','decoder.','context.history.']};assert all(changed.values());detail['changed_tensor_counts_joint']=changed
  item[arm]=detail
 assert item['learned']['evaluation_protocol']==item['uniform']['evaluation_protocol'];rows.append(item)
result={'status':'PASS','device':'CPU only','training_jobs_modified':False,'all_five_pairs':rows}
print(json.dumps(result,indent=2))
