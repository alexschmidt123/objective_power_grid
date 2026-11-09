"""Read-only CPU audit of three completed MoE experiments;no model execution."""
import os,sys,pathlib,json,hashlib,time,collections,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np,torch
from scipy.stats import t as student_t
torch.set_num_threads(2)
R=pathlib.Path.home()/'Documents/objective_power_grid_temp/scratch_runs';parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=pathlib.Path,default=R/'moe_hec_runtime_efficiency_audit_20261109');args=parser.parse_args();A=args.output.expanduser().resolve();assert A.is_relative_to(R.resolve()) and A!=R.resolve(),'Audit output must be a new scratch directory';A.mkdir(exist_ok=False)
NAMES={'collaborative':'moe_collaborative_soft_queue_20261108','sg':'moe_specialization_guided_eig_20261108','hec':'moe_hec_sboed_20261108'};inputs={};payload={};summary={};begin=time.time()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):
 inputs[str(p)]=sha(p);v=json.loads(p.read_text());payload[str(p.relative_to(R))]=v;return v
def stats(v):
 n=len(v);m=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(n));q=float(student_t.ppf(.975,n-1));return dict(n=n,mean=m,MCSE=se,CI95=[m-q*se,m+q*se],p_two_sided=float(2*student_t.sf(abs(m/max(se,1e-300)),n-1)))
for label,name in NAMES.items():
 p=R/name;s={};s['config']=read(p/'CONFIG_LOCKED.json');s['completion']=read(p/'evidence/EXPERIMENT_COMPLETION.json');s['evaluation']=read(p/'evaluation/COMPLETION.json');s['training_started']=read(p/'evidence/TRAINING_STARTED.json');s['preparation']=read(p/'preparation/RESULT.json') if (p/'preparation/RESULT.json').exists() else None;s['arms']={}
 for arm in ['control','candidate']:
  result=read(p/f'arms/{arm}/RESULT.json');start=read(p/f'arms/{arm}/STARTED.json');lp=p/f'TRAINING_LOGS/{arm}_metrics.jsonl';inputs[str(lp)]=sha(lp);rows=[json.loads(x) for x in lp.read_text().splitlines()];payload[str(lp.relative_to(R))]=rows;u=lambda row:row.get('continuation_updates',row.get('native_iterations'));blocks=[]
  for a,b in zip(rows,rows[1:]):
   dn=u(b)-u(a)
   if dn>0:blocks.append(dict(first=u(a),last=u(b),seconds_per_update=(b['seconds']-a['seconds'])/dn,updates=dn))
  n=u(rows[-1]);assert n==result.get('continuation_updates',result.get('native_iterations'));assert rows[-1]['native_optimizer_steps']==result['native_optimizer_steps'];expected=10001 if label=='collaborative' else (12001 if label=='sg' else 14001)
  cp=p/f'arms/{arm}/actor.pt';runtime=p/f'arms/{arm}/runtime.pkl';inputs[str(cp)]=sha(cp);inputs[str(runtime)]=sha(runtime);assert inputs[str(cp)]==result['actor_sha256'];assert inputs[str(runtime)]==result['runtime_sha256'];loaded=torch.load(cp,map_location='cpu',weights_only=True);iteration=loaded.get('native_iteration',loaded.get('native_iterations'));assert iteration==expected
  state=loaded['actor_state'];groups=collections.defaultdict(int)
  for k,v in state.items():groups[k.split('.')[0]]+=v.numel()
  metadata={k:v for k,v in loaded.items() if k!='actor_state'};metadata={k:v for k,v in metadata.items() if isinstance(v,(str,int,float,bool,type(None)))}
  r=dict(result=result,start=start,log_rows=len(rows),first=rows[0],last=rows[-1],mean_seconds_per_update=result['seconds']/n,median_logged_block_seconds_per_update=float(np.median([b['seconds_per_update'] for b in blocks])),blocks=blocks,terminal_IO_and_unallocated_arm_seconds=result['seconds']-rows[-1]['seconds'],checkpoint=dict(sha256=inputs[str(cp)],runtime_sha256=inputs[str(runtime)],bytes=cp.stat().st_size,runtime_bytes=runtime.stat().st_size,metadata=metadata,state_elements_by_prefix=dict(groups),total_state_elements=sum(groups.values()),tensor_devices=sorted({str(v.device) for v in state.values()})))
  if label=='hec':
   n200=next(v for v in rows if u(v)==200);r['phase_B']=dict(updates=200,seconds=n200['seconds'],seconds_per_update=n200['seconds']/200);r['phase_C']=dict(updates=1800,seconds=rows[-1]['seconds']-n200['seconds'],seconds_per_update=(rows[-1]['seconds']-n200['seconds'])/1800);r['first_phase_C_logged']=next(v for v in rows if u(v)>200)
  s['arms'][arm]=r;del loaded,state
 s['pair_training_seconds']=sum(v['result']['seconds'] for v in s['arms'].values());s['prep_seconds']=(s['preparation'] or {}).get('seconds',0);s['unallocated_driver_seconds']=s['completion']['seconds']-s['pair_training_seconds']-s['evaluation']['seconds']-s['prep_seconds'];s['evaluation_receipts']={}
 for f in (p/'evaluation').glob('*_receipt.json'):s['evaluation_receipts'][f.name]=read(f)
 s['rollout_observations_in_receipts']=sum(v.get('simulator_observations',0) for v in s['evaluation_receipts'].values());s['receipt_seconds_sum']=sum(v.get('seconds',0) for v in s['evaluation_receipts'].values());s['eig']=read(p/'evaluation/EIG_RESULTS.json');s['specialization']=read(p/'evaluation/SPECIALIZATION_RESULTS.json')
 for n in ['COMMANDER_RESULTS.json','ROUTING_RESULTS.json','COMPETENCY_RESULTS.json','MECHANISM_RESULTS.json','CONDITIONAL_COMPETENCE.json','LATENCY.json','INFERENCE_LATENCY.json']:
  if (p/'evaluation'/n).exists():s[n]=read(p/'evaluation'/n)
 for n in ['PREPARATION_DIRECT_PROVIDER_COUNTS.json','FINAL_PRESERVATION.json','CHECKS.json']:
  if (p/'evidence'/n).exists():s[n]=read(p/'evidence'/n)
 source_files=['run_experiment.py','train_pair.py' if label=='hec' else 'train_sg_pair.py' if label=='sg' else 'train_soft_pair.py','evaluate_hec.py' if label=='hec' else 'evaluate_sg.py' if label=='sg' else 'evaluate_soft.py']
 if label=='hec':source_files+=['hec_policy.py','specialist_tasks.py','migration.py','prepare_specialists.py']
 for name0 in source_files:
  f=p/'source'/name0;inputs[str(f)]=sha(f);payload[str(f.relative_to(R))]=f.read_text()
 summary[label]=s
# Same-bank paired differences from preserved per-case scores, no new rollouts.
def scores(label,method):
 f=R/NAMES[label]/f'evaluation/{method}_scores.npy';inputs[str(f)]=sha(f);return np.load(f)
comparison={}
for label,parent in [('sg','parent'),('hec','sg')]:
 sc=scores(label,parent);q={}
 for arm in ['control','candidate']:
  v=scores(label,arm)-sc;st=stats(v);cost=(summary[label]['arms'][arm]['result']['seconds']+summary[label]['prep_seconds'])/3600;assert len(sc)==2000;q[arm]=dict(gain_vs_same_bank_frozen_parent=st,incremental_training_plus_prep_GPU_wall_hours=cost,conditional_gain_nats_per_GPU_wall_hour=st['mean']/cost,rate_CI95=[z/cost for z in st['CI95']],episodes=2000000,interpretation='warm-start-conditional endpoint gain per observed incremental training time;not a learning-curve derivative or cross-bank ranking')
 q['candidate_minus_control']=stats(scores(label,'candidate')-scores(label,'control'));comparison[label]=q
# Independent MCSE labels are not paired intervals; match stored primary exactly.
for label,q in comparison.items():
 stored=summary[label]['eig']['primary_candidate_minus_control'];assert abs(q['candidate_minus_control']['mean']-stored['mean'])<1e-12
summary['conditional_efficiency']=comparison
c=summary['collaborative']['arms']['candidate']['result'];s=summary['sg']['arms']['candidate']['result'];h=summary['hec']['arms']['candidate']['result'];summary['lineage']=dict(candidate_only_training_seconds=[c['seconds'],c['seconds']+s['seconds'],c['seconds']+s['seconds']+h['seconds']+summary['hec']['prep_seconds']],candidate_native_counter=[10001,12001,14001],candidate_real_episodes=[10021000,12021000,14021000],all_three_two_arm_driver_seconds=sum(summary[k]['completion']['seconds'] for k in NAMES))
# Verify every read original byte-for-byte after CPU audit.
for path,hsh in inputs.items():assert sha(pathlib.Path(path))==hsh,path
(A/'RAW_RECEIPTS_AND_LOGS.json').write_text(json.dumps(payload,indent=2));(A/'AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2));(A/'INPUT_PRESERVATION.json').write_text(json.dumps(dict(status='VERIFIED',files=inputs,files_checked=len(inputs),scientific_training_updates=0,neural_inference_calls=0,new_simulator_calls=0,GPU_used=False,torch_version=torch.__version__,seconds=time.time()-begin,actor_checkpoints_read_CPU_only=True,runtimes_hashed_not_deserialized=True),indent=2));print(json.dumps(dict(status='COMPLETE',output=str(A),input_files_verified=len(inputs),scientific_updates=0,neural_inference_calls=0,simulator_calls=0,seconds=time.time()-begin)))
