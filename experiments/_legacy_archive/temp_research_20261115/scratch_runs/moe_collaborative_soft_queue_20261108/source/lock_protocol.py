"""Final pretrain lock after mandatory upstream completion and GPUpreflight."""
from native_common import *
assert not (O/'PRETRAIN_MANIFEST.json').exists();queue=json.loads((O/'QUEUE_MANIFEST.json').read_text())
for n,h in queue['files'].items():assert sha(O/n)==h,n
for n in ['SOFT_TEST_RESULTS.json','NATIVE_LOOP_TEST.json','EVALUATION_TEST_RESULTS.json','GPU_PREFLIGHT.json']:assert json.loads((O/'evidence'/n).read_text())['status']=='PASS'
roots=[R/name for name in ['moe_conditional_strategy_audit_20261105','moe_posterior_validation_recovery_20261106','moe_independent_posterior_reference_20261106','moe_geometry_adapted_reference_20261106','moe_proximity_stratified_moments_20261106','moe_reference_closure_and_eig_plan_20261106','moe_competitive_action_routing_pilot_20261107','moe_option_specialization_20261107','moe_information_specialized_option_20261107','moe_task_specialized_option_20261107']];before={str(root):{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()} for root in roots};put(O/'evidence/ORIGINALS_BEFORE.json',before);refs={str(p):sha(p) for p in (P/'source').rglob('*') if p.is_file()};prov=json.loads((O/'evidence/INPUT_PROVENANCE.json').read_text())
for v in prov['copied_checkpoints'].values():assert sha(v['path'])==v['sha256'];refs[v['path']]=v['sha256']
refs[str(PARENT)]=sha(PARENT);put(O/'evidence/REFERENCE_SOURCE_INPUTS.json',refs);files=dict(queue['files']);files['QUEUE_MANIFEST.json']=sha(O/'QUEUE_MANIFEST.json')
for directory in ['inputs','evidence']:
 for p in (O/directory).glob('*'):
  if p.is_file() and (directory=='inputs' or p.name in ['SOFT_TEST_RESULTS.json','NATIVE_LOOP_TEST.json','EVALUATION_TEST_RESULTS.json','GPU_PREFLIGHT.json','REFERENCE_REUSE.json']):files[str(p.relative_to(O))]=sha(p)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_TRAINING',time=time.time(),initialization='freshnativeCUDAseed101;independentsoftactorexpertseeds;emptyAdam/replay',files=files,historical_roots=len(roots),historical_files=sum(len(v) for v in before.values()),scientific_optimizer_updates=0));print('SOFT_PROTOCOL_LOCKED')
