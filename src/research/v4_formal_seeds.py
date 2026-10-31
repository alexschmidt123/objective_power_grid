"""Formal V4 seeds101/102: frozen utility mechanism, native four-hour allowance."""
import time
ENTRY=time.monotonic()
import os,sys,json,csv,hashlib,copy,subprocess,fcntl,traceback
from pathlib import Path
import torch,numpy as np
from src.research import v4_utility_seed1 as base
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.v4_utility_routing import paired_credit,add_credit
from src.research.engine import rollout
from src.research.benchmarks import make_benchmark
from src.research.compute_budget import ComputeClock
from src.research.journal_validation import CommonValidation
R=Path(__file__).resolve().parents[2];F=R/'scratch_runs/moe_sboed_v4_formal_seeds101_102';write=base.write;sha=base.sha;stat=base.stat

def prep():
 F.mkdir(exist_ok=False)
 for d in ['reports','logs','provenance','seed101','seed102']:(F/d).mkdir()
 old=base.config();archive=R.parent/'objective_power_grid/experiments/_legacy_archive/labpc';arrays=archive/'objective_power_grid_matched_compute_20261020/experiments/10202026/10202026_icml_matched_compute_protocol/benchmarks/location_seed101_matched_compute/evaluations';cp102=R/'scratch_runs/moe_sboed_v4_corrected_v1_finetune/seed102_historical_selected.pt'
 assert sha(arrays/'location_cases.npz')=='cbd2632af41a8c6ca8cbe55066d77b98791565b7ac301b0bea1965e84ec50f5b';assert sha(arrays/'location_contrasts.npy')=='703671b0fb01ddf74dfe60d51e44914c71647c92e7e5ee51d111a9acfcecfde8';assert sha(cp102)=='1256720d370f42ab7a069ffa87a20a565bcd3623d60480b377c9d02c9e79a65c'
 for seed in [101,102]:
  c=F/f'seed{seed}';[(c/d).mkdir() for d in ['reports','logs','checkpoints','evaluations','provenance']];cfg=copy.deepcopy(old);cfg.update(method='moe_sboed_v4_formal',formal_not_pilot=True,seed=seed,budget_seconds=14400,native_maximum_updates=50000,initialization='FROM_SCRATCH_NO_HISTORICAL_WEIGHTS',development_seed=395000+seed,diagnostic_seed=395100+seed,historical_arrays_directory=str(arrays),historical_cases_sha256=sha(arrays/'location_cases.npz'),historical_bank_sha256=sha(arrays/'location_contrasts.npy'),historical_reported_EIG=11.04404166 if seed==101 else 10.83415593,hourly_checkpoints_seconds=[3600,7200,10800,14400],contextual_checkpoint=old['contextual_checkpoint'] if seed==101 else str(cp102),contextual_sha256=old['contextual_sha256'] if seed==101 else sha(cp102),final_seed=1001,official_seed1001_unused='Used only after final checkpoint freeze, per formal request; never for development or selection')
  write(c/'configuration.json',cfg);write(c/'status.json',dict(state='QUEUED',seed=seed,updates=0,charged_seconds=0,formal_not_pilot=True))
 # Reuse unchanged, already passing native mechanism tests; no new training pilot.
 tests={}
 for name in ['cpu_tests','gpu_smoke']:
  p=base.C/'provenance'/f'{name}.json';x=json.loads(p.read_text());assert x['status']=='PASS';tests[name]=dict(receipt=str(p),sha256=sha(p),status='PASS')
 states=[]
 for seed in [101,102]:
  torch.manual_seed(seed);m=DeepSeekPolicy();assert m.counts()['total_parameters']==8434 and m.counts()['activated_parameters_per_context']==6898;n=DeepSeekPolicy();n.load_state_dict(m.state_dict(),strict=True);assert all(torch.equal(x,y) for x,y in zip(m.state_dict().values(),n.state_dict().values()));states.append(torch.cat([p.detach().flatten() for p in m.parameters()]))
 assert not torch.equal(states[0],states[1]);write(F/'provenance/essential_checks.json',dict(status='PASS',core_tests=tests,formal_seed_fresh_initialization=True,strict_state_dict_compatibility=True,distinct_initial_states=True,native_max_updates=50000,per_seed_cap=14400,no_pilot_cap=True,utility_source_sha256=sha(R/'src/research/v4_utility_routing.py'),architecture_source_sha256=sha(R/'src/research/deepseek_sboed.py')))
 docs={
 'FORMAL_V4_PROTOCOL.md':'Formal sequential seeds101then102, from scratch. Each seed: native maximum50000updates OR14400charged seconds, with serialization reserve; no8192/1h/2hcap, no early stopping by development EIG. B2000,L2000,T30,Adam5e-5 betas(.8,.998),.98LRdecay every1000updates,clip5,balance.001. Wait normal GPU allocation; never preempt existing seed1 or other jobs. Evaluation/diagnostics after each frozen final; no approvals between seeds.',
 'V1_ARCHITECTURE_EQUIVALENCE.md':'Exactly pinned DeepSeekPolicy:8434trainable/6898active,1shared+7routed width8SwiGLU,top3,16Dhistory,original residual/emitter/feasibility and raw selected softmax weights. No new inference parameters. Existing mechanism CPU/nativeGPU checks passed; formal seed fresh-state compatibility additionally passed.',
 'V4_SPECIALIZATION_METHOD.md':(base.C/'reports/V4_UTILITY_ROUTING_METHOD.md').read_text()+'\nThe formal mechanism is identical to seed1; no loss/coefficient tuning. Every100updates, B64,L1024 controlled one-swap paired full-horizon continuations; detached RMS-bounded logistic ranking, additive router gradient capped at5% primary router norm. All other parameters retain original full-loss gradients. Numerical safeguards logged;10consecutive invalid credit events stop as numerical/mechanism failure rather than silently train V1.',
 'SEED101_TRAINING.md':'QUEUED:14400seconds, native ceiling50000. No optimizer update yet.',
 'SEED102_TRAINING.md':'QUEUED after seed101 training/final evaluation/diagnostics. Same14400seconds/native ceiling50000; no additional approval needed.',
 'FORMAL_COMPUTE_ACCOUNTING.md':'Per seed14400charged seconds includes initialization, applicable existing GPU-preparation receipt, every primary/auxiliary simulation/backward pass, logging/checkpoint IO. Development/final evaluation separately metered; no trainable phase excluded. Resume retains spent credit; failed/interrupted work is not granted another4h. Atomic manifests store optimizer/scheduler/RNG/checkpoint hashes.',
 'FORMAL_EIG_RESULTS.md':'PENDING; no formal V4 score is available.',
 'PAIRED_V1_COMPARISON.md':'Final checkpoint only:2000paired cases/1Mcontrasts from exact historical canonical arrays, verifiedSHA256. Evaluate V4 seed101 against contextualV1 seed101; seed102 against contextualV1 seed102. No claim if pairedCI contains0; one seed is not general superiority.',
 'EXPERT_SPECIALIZATION_ANALYSIS.md':'PENDING. Frozen random-top3 preserves original raw gate semantics; individual and whole-routed masks are diagnostics with distribution-shift limitations. Independent-history conditional one-swap utilities and cross-seed summaries do not establish semantic expert roles alone.',
 'FINAL_SCIENTIFIC_DECISION.md':'PENDING both formal runs and paired evaluations. Negative results preserved; no automatic promotion or extra training.'}
 for name,text in docs.items():(F/'reports'/name).write_text('# '+name.replace('.md','')+'\n\n'+text+'\n')
 table();print('FORMAL PREPARED',F,flush=True)

def table():
 fields=['seed','status','checkpoint','checkpoint_sha256','total_charged_seconds','optimizer_updates','training_trajectories','training_contrasts','EIG','MC_SE','historical_V1_reported_EIG','historical_V1_paired_EIG','paired_difference','paired_MC_SE','paired_CI_low','paired_CI_high','native_schedule_completion','specialization_evidence'];rows=[]
 for seed in [101,102]:
  c=F/f'seed{seed}';s=json.loads((c/'status.json').read_text());cfg=json.loads((c/'configuration.json').read_text());cp=s.get('latest_checkpoint') or {};r=dict.fromkeys(fields,'');r.update(seed=seed,status=s['state'],checkpoint=cp.get('path',''),checkpoint_sha256=cp.get('sha256',''),total_charged_seconds=s.get('charged_seconds',0),optimizer_updates=s.get('update',s.get('updates',0)),training_trajectories=s.get('fresh_training_trajectories',0),training_contrasts=2000,historical_V1_reported_EIG=cfg['historical_reported_EIG'],native_schedule_completion=s.get('update',0)>=50000,specialization_evidence=s.get('active_specialization_updates',0))
  p=c/'evaluations/formal_paired_comparison.json'
  if p.exists():
   x=json.loads(p.read_text());a=x['v4_minus_contextual_v1'];r.update(EIG=x['v4']['mean'],MC_SE=x['v4']['MC_SE'],historical_V1_paired_EIG=x['contextual_v1']['mean'],paired_difference=a['mean'],paired_MC_SE=a['MC_SE'],paired_CI_low=a['CI95'][0],paired_CI_high=a['CI95'][1])
  rows.append(r)
 p=F/'formal_results.csv';tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 tmp.replace(p);return rows

def formal_eval(m,cfg,cp,c):
 d=Path(cfg['historical_arrays_directory']);assert sha(d/'location_cases.npz')==cfg['historical_cases_sha256'] and sha(d/'location_contrasts.npy')==cfg['historical_bank_sha256'];cases=np.load(d/'location_cases.npz');bank=np.load(d/'location_contrasts.npy');assert len(bank)==1000000 and len(cases['truth'])==2000
 val=CommonValidation(cfg['benchmark'],30,1001,2000,1000000,batch=16,chunk=16384,device='cuda:0');val.truth=torch.as_tensor(cases['truth'],device='cpu');val.noise=torch.as_tensor(cases['noise'],device='cpu');val.bank=torch.as_tensor(bank,device='cpu');val.identity=hashlib.sha256(val.truth.numpy().tobytes()+val.noise.numpy().tobytes()+val.bank.numpy().tobytes()).hexdigest();assert sha(cfg['contextual_checkpoint'])==cfg['contextual_sha256'];ref=load_deepseek(cfg['contextual_checkpoint'],'cuda:0');results={};scores={}
 for name,p in [('v4',m),('contextual_v1',ref)]:
  a,meta=val(p);scores[name]=a;results[name]=dict(**stat(a),metadata=meta);np.save(c/'evaluations'/f'formal_{name}_scores.npy',a)
 results['v4_minus_contextual_v1']=stat(scores['v4']-scores['contextual_v1']);results['checkpoint']=cp;results['cases_sha256']=cfg['historical_cases_sha256'];results['bank_sha256']=cfg['historical_bank_sha256'];results['seed']=cfg['seed'];results['decision']='PAIRED_GAIN_FOR_THIS_TRAINING_SEED' if results['v4_minus_contextual_v1']['CI95'][0]>0 else 'NEGATIVE_FOR_THIS_TRAINING_SEED' if results['v4_minus_contextual_v1']['CI95'][1]<0 else 'INCONCLUSIVE_FOR_THIS_TRAINING_SEED';write(c/'evaluations/formal_paired_comparison.json',results)

# Lease is acquired by the sequential controller and inherited by this worker.
def train(seed,resume=None):
 c=F/f'seed{seed}';cfg=json.loads((c/'configuration.json').read_text());base.C=c
 fds=[int(x) for x in os.environ.get('V4_FORMAL_LOCK_FDS','').split(',') if x];assert len(fds)==3,'No inherited exclusive device lease';leases=[os.fdopen(os.dup(fd),'a+') for fd in fds]
 torch.set_num_threads(4);torch.cuda.init();torch.manual_seed(seed);m=DeepSeekPolicy().cuda().train();o=torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998));sch=torch.optim.lr_scheduler.ExponentialLR(o,.98);b=make_benchmark(cfg['benchmark']);manifest=json.loads((F/'provenance/source_sha256.json').read_text());smoke=json.loads((R/'scratch_runs/moe_sboed_v4_utility_routing_seed1/provenance/gpu_smoke.json').read_text());credit=smoke['seconds'];u=0;aux_seconds=0.;active=0;invalid=0;bound=max(1.,smoke['native_full_update_seconds']);latest=None
 if resume:
  p=Path(resume);receipt=json.loads(p.with_suffix('.json').read_text());assert sha(p)==receipt['sha256'];saved=torch.load(p,map_location='cuda:0',weights_only=False);assert saved['seed']==seed and saved['configuration_sha256']==sha(c/'configuration.json') and saved['source_manifest_sha256']==sha(F/'provenance/source_sha256.json');m.load_state_dict(saved['state_dict'],strict=True);o.load_state_dict(saved['optimizer']);sch.load_state_dict(saved['scheduler']);u=saved['update'];aux_seconds=saved['auxiliary_seconds'];active=saved['active_specialization_updates'];torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);old=json.loads((c/'provenance/compute_clock.json').read_text());end=old.get('stopped_at',time.monotonic());credit=max(receipt['charged_seconds'],old['credit_seconds']+end-old['start']-sum(bb-aa for aa,bb,reason in old['excluded']));assert credit<14400;oldpath=c/'provenance/compute_clock.json';oldpath.rename(oldpath.with_name('compute_clock_before_resume_'+str(int(time.time()))+'.json'))
 elif (c/'provenance/started.json').exists():raise RuntimeError('Started seed requires explicit reviewed checkpoint resume')
 clock=ComputeClock(c/'provenance/compute_clock.json',14400,credit_seconds=credit,hardware=torch.cuda.get_device_name(0),start=ENTRY);next_recovery=clock.spent()+300;next_hour=next((t for t in [3600,7200,10800,14400] if t>clock.spent()),14400);write(c/'provenance/started.json',dict(PID=os.getpid(),seed=seed,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),configuration_sha256=sha(c/'configuration.json'),source_manifest_sha256=sha(F/'provenance/source_sha256.json'),GPU=torch.cuda.get_device_name(0),from_scratch=resume is None,resume_checkpoint=resume,prior_credit_seconds=credit))
 def status(state,**kw):
  data=dict(state=state,seed=seed,PID=os.getpid(),update=u,charged_seconds=clock.spent(),fresh_training_trajectories=u*2000,latest_checkpoint=latest,active_specialization_updates=active,auxiliary_seconds=aux_seconds,**kw);write(c/'status.json',data);(F/'reports'/f'SEED{seed}_TRAINING.md').write_text('# Formal seed'+str(seed)+'\n\n'+json.dumps(data,indent=2))
 def save(label):
  nonlocal latest
  for rel,h in manifest.items():assert sha(R/rel)==h,'Frozen source changed: '+rel
  p=c/'checkpoints'/f'update{u}_{label}_{int(time.time())}.pt';tmp=p.with_suffix('.partial');torch.save(dict(policy=copy.deepcopy(m).cpu(),state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},optimizer=o.state_dict(),scheduler=sch.state_dict(),seed=seed,update=u,configuration=cfg,source_manifest_sha256=sha(F/'provenance/source_sha256.json'),configuration_sha256=sha(c/'configuration.json'),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),auxiliary_seconds=aux_seconds,active_specialization_updates=active),tmp);tmp.replace(p);h=sha(p);clock.checkpoint(str(p));row=dict(path=str(p),sha256=h,update=u,seed=seed,charged_seconds=clock.spent(),eligible=clock.spent()<=14400);write(p.with_suffix('.json'),row)
  if row['eligible']:latest=row;write(c/'checkpoints/latest_eligible.json',row)
 save('initial' if not resume else 'resumed');status('FORMAL_TRAINING_RUNNING')
 try:
  while u<50000 and clock.can_start(max(30.,2*bound)):
   t=time.monotonic();u+=1;o.zero_grad(set_to_none=True)
   with torch.device('cuda:0'):
    r=rollout(b,m,30,2000,seed+u*104729,2000);info=r['info'][:,-1];value=float(info.mean().detach());sat=float((info>np.log(2001)-.1).double().mean());loss=-info.mean()+m.balance_loss();assert torch.isfinite(loss);loss.backward();routing=m.routing_statistics() if u==1 or u%25==0 else None;del r,info,loss;m._gate_cache=[];aux=None
    if u%100==0:
     start=time.monotonic();pairseed=39600000+seed*100000+u;record=paired_credit(b,m,pairseed,64,1024);aux=add_credit(m,record,.05);torch.cuda.synchronize();elapsed=time.monotonic()-start;aux_seconds+=elapsed;active+=int(aux['applied']);invalid=0 if aux['applied'] else invalid+1;aux.update(update=u,seed=seed,seconds=elapsed,randomness_seed=pairseed,same_prefix_verified=True,paired_randomness_sha256=hashlib.sha256(record['truth'].cpu().numpy().tobytes()+record['noise'].cpu().numpy().tobytes()+record['particles'].cpu().numpy().tobytes()).hexdigest())
     with (c/'logs/utility_credit.jsonl').open('a') as f:f.write(json.dumps(aux)+'\n')
     if invalid>=10:raise FloatingPointError('10consecutive invalid/zero utility corrections; specialization not silently disabled')
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);o.step()
    if (u-1)%1000==0:sch.step()
    torch.cuda.synchronize()
   bound=max(bound,time.monotonic()-t)
   if u==1 or u%25==0:
    row=dict(update=u,seed=seed,charged_seconds=clock.spent(),training_sPCE=value,saturation_fraction=sat,gradient_norm=float(gn),learning_rate=o.param_groups[0]['lr'],routing=routing,auxiliary_seconds=aux_seconds,auxiliary=aux,active_specialization_updates=active)
    with (c/'logs/training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True);status('FORMAL_TRAINING_RUNNING')
   milestone=u in [1000,3000,5000] or (u>=10000 and u%5000==0);hour_due=clock.spent()>=next_hour
   if u==1 or milestone or hour_due or clock.spent()>=next_recovery:
    save('charged_hour'+str(next_hour//3600) if hour_due else 'milestone' if milestone else 'recovery');next_recovery=clock.spent()+300
    if hour_due:next_hour=next((x for x in [3600,7200,10800,14400] if x>clock.spent()),14401)
   if milestone:
    status('DEVELOPMENT_MONITORING')
    with clock.exclude('validation'):base.development(m,cfg,u)
    m.train();status('FORMAL_TRAINING_RUNNING')
  save('final');end=clock.stop();assert latest;write(c/'provenance/training_complete.json',dict(seed=seed,clock=end,updates=u,fresh_trajectories=u*2000,contrasts=2000,auxiliary_seconds=aux_seconds,active_specialization_updates=active,stop_reason='NATIVE_50000_COMPLETE' if u>=50000 else 'FOUR_HOUR_COMPUTE_LIMIT',latest=latest));status('TRAINING_COMPLETE_FINAL_EVALUATION');assert sha(latest['path'])==latest['sha256'];m=torch.load(latest['path'],map_location='cuda:0',weights_only=False)['policy'];formal_eval(m,cfg,latest,c);status('SPECIALIZATION_DIAGNOSTICS');base.diagnostics(m,cfg);status('FORMAL_SEED_COMPLETE')
 except Exception as e:
  write(c/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),update=u,latest=latest,clock=clock.stop()));status('FAILED_REQUIRES_REVIEW');raise

def acquire():
 files=[]
 try:
  for p in [R.parent/'objective_power_grid/.gpu0.lock',Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock']:
   f=p.open('a+');files.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if busy:raise BlockingIOError('GPU compute process '+busy)
  return files
 except Exception:
  for f in files:f.close()
  raise

def summary():
 rows=table();texts=[]
 for seed in [101,102]:
  p=F/f'seed{seed}/evaluations/formal_paired_comparison.json'
  if p.exists():texts.append(json.loads(p.read_text()))
 (F/'reports/FORMAL_EIG_RESULTS.md').write_text('# Formal results\n\n'+json.dumps(rows,indent=2));(F/'reports/PAIRED_V1_COMPARISON.md').write_text('# Matched historical V1 comparisons\n\n'+json.dumps(texts,indent=2))
 if len(texts)==2:
  gains=[x['v4_minus_contextual_v1']['CI95'][0]>0 for x in texts];decision='PAIRED_GAIN_BOTH_SEEDS; generality still requires independent replication' if all(gains) else 'NO_REPLICATED_PAIRED_IMPROVEMENT';(F/'reports/FINAL_SCIENTIFIC_DECISION.md').write_text('# Scientific decision\n\n'+decision+'\nNo automatic STABLE promotion or further seeds. Individual seed results and limitations remain in paired comparison.\n')
 diagnostics={}
 for seed in [101,102]:
  p=F/f'seed{seed}/evaluations/specialization_diagnostics.json'
  if p.exists():diagnostics[str(seed)]=dict(path=str(p),sha256=sha(p),frozen_interventions=json.loads(p.read_text())['frozen_interventions'])
 if diagnostics:(F/'reports/EXPERT_SPECIALIZATION_ANALYSIS.md').write_text('# Cross-seed frozen diagnostics\n\n'+json.dumps(diagnostics,indent=2)+'\nDifferent interventions/weights are not proof of causal semantic roles. No retrained dense or no-credit controls in this formal pair.\n')
 computes={}
 for seed in [101,102]:
  p=F/f'seed{seed}/provenance/training_complete.json'
  if p.exists():computes[str(seed)]=json.loads(p.read_text())
 if computes:(F/'reports/FORMAL_COMPUTE_ACCOUNTING.md').write_text('# Formal charged compute\n\n'+json.dumps(computes,indent=2))

def queue(resume_seed=None,resume_checkpoint=None):
 lock=(F/'provenance/controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (F/'provenance/controller_started.json').exists():
  if resume_seed is None:raise RuntimeError('Controller already started; inspect before resume')
  (F/'provenance/controller_started.json').rename(F/'provenance'/('controller_before_resume_'+str(int(time.time()))+'.json'))
 write(F/'provenance/controller_started.json',dict(PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 def status(state,**kw):write(F/'status.json',dict(state=state,PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
 for seed in [101,102]:
  if resume_seed is not None and seed<resume_seed:continue
  c=F/f'seed{seed}'
  if json.loads((c/'status.json').read_text())['state']=='FORMAL_SEED_COMPLETE':continue
  while True:
   try:lease=acquire();break
   except (BlockingIOError,RuntimeError) as e:status('WAITING_GPU',next_seed=seed,reason=str(e),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip());time.sleep(30)
  for rel,h in json.loads((F/'provenance/source_sha256.json').read_text()).items():assert sha(R/rel)==h,'Frozen source changed: '+rel
  fds=[f.fileno() for f in lease];env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None);env['V4_FORMAL_LOCK_FDS']=','.join(map(str,fds));cmd=['bash','-c','source ./run.sh; exec python3 -m src.research.v4_formal_seeds train "$1"','v4-formal',str(seed)]
  if resume_seed==seed:
   cmd.append(str(resume_checkpoint))
  logpath=F/'logs'/(f'seed{seed}.log' if resume_seed!=seed else f'seed{seed}_resume_{int(time.time())}.log')
  with logpath.open('x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(fds),start_new_session=True)
  for f in lease:f.close()
  status('FORMAL_SEED_RUNNING',seed=seed,child_PID=child.pid,log=str(logpath))
  while child.poll() is None:summary();time.sleep(30)
  exitfile=F/'provenance'/f'seed{seed}_exit.json'
  if exitfile.exists():exitfile.rename(exitfile.with_name(f'seed{seed}_exit_before_resume_{int(time.time())}.json'))
  write(exitfile,dict(returncode=child.returncode));summary()
  if child.returncode:status('FAILED_REQUIRES_REVIEW',seed=seed,returncode=child.returncode);return
 status('FORMAL_BOTH_SEEDS_COMPLETE');summary()
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='prepare':prep()
 elif phase=='queue':queue()
 elif phase=='resume':queue(int(sys.argv[2]),sys.argv[3])
 elif phase=='train':train(int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else None)
 else:raise ValueError(phase)
