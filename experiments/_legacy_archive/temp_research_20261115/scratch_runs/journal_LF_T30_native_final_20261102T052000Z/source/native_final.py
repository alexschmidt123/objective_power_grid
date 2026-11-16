import pathlib,os,sys,time,json,datetime,csv,hashlib,subprocess,fcntl,signal,traceback,math,shutil
R=pathlib.Path.home()/'Documents/objective_power_grid_temp';os.chdir(R);sys.path.insert(0,str(R));PY=pathlib.Path.home()/'miniconda3/envs/mocu_optimized/bin/python'
C=pathlib.Path(__file__).parent.parent
METHODS=['random','fixed','dad','rl_sboed','vsoed','scorebed'];SEEDS=list(range(101,106))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def put(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False));q.replace(p)
def table(p,rows,fields=None):
 if not rows:return
 q=p.with_suffix('.tmp')
 with q.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
 q.replace(p)
def verify():
 for f,h in read(C/'source_manifest.json')['sha256'].items():
  assert sha(R/f)==h,'Native source changed '+f
 for f,h in read(C/'frozen_files.json').items():assert sha(C/f)==h,'Cohort source/config changed '+f
def locks():
 from src.research.location_baseline_campaign import acquire
 return acquire()
def prepare():
 import torch,numpy as np,yaml
 from src.research.location_baseline_campaign import config
 from src.research.runner import freeze_source
 from src.research.journal_validation import CommonValidation
 from src.research.benchmarks import make_benchmark
 from src.research.journal_checkpoints import load_checkpoint
 from src.research.engine import rollout
 torch.set_num_threads(4)
 for d in ['configs','logs','runs','readiness','banks','evaluations','reports']:(C/d).mkdir(exist_ok=True)
 freeze_source(R,C)
 raw=config('dad',43200);physical={k:v for k,v in raw.items() if k!='research'};physical['research']={'paper_protocol':{'corrected_methods':True}};put(C/'physical.json',physical)
 banks={}
 for name,seed in [('development',2026110205150101),('sealed_final',2026110205150102)]:
  v=CommonValidation(physical,30,seed,2000,1000000);p=C/'banks'/f'{name}.npz';np.savez(p,truth=v.truth.numpy(),noise=v.noise.numpy(),contrasts=v.bank.numpy())
  banks[name]=dict(seed=seed,systems=2000,contrasts=1000000,path=str(p),file_sha256=sha(p),draws_sha256=v.identity,access='SEALED_NO_POLICY_EVALUATION_UNTIL_USER_FREEZES_MOE' if name=='sealed_final' else 'DEVELOPMENT_ONLY')
 put(C/'bank_manifest.json',banks)
 for method in METHODS:
  cfg=config(method,43200);p=cfg['research']['journal_protocol'];p['study_id']=C.name;p['validation'].update(seed=2026110205150103,systems=64);p['stopping_rule']['maximum_training_seconds']=43200;p['compute_budget'].update(seconds=43200,checkpoint_seconds=3600,finalization_reserve_seconds=120);p['final_evaluation']['seed']=banks['sealed_final']['seed'];p['evaluation_seeds']=[banks['sealed_final']['seed']]
  p['checkpoint_selection_mode']='final_compute_budget'
  s=p['method_settings'][method] if method!='random' else {}
  if method=='fixed':s.update(batch_size=2000,contrasts=2000,learning_rate=1e-4,spce_gradient_clip=None,spce_lr_decay=.98,reference_optimizer_updates=50000,reference_training_episodes=100000000,reference_schedule_episodes=100000000)
  if method=='rl_sboed':s['reference_schedule_episodes']=2000000
  if method=='scorebed':
   s.update(scorebed_restarts=5,scorebed_reference_sampler=False,scorebed_native_reference_schedule=True,score_schedule_reference_steps=50000,policy_schedule_reference_steps=20000,scorebed_declared_full_score_updates=50000,scorebed_declared_full_policy_updates=100000,scorebed_pretraining_episodes=50000*1024,reference_schedule_episodes=10054+150000*1024)
   s.pop('scorebed_compute_phase_plan',None)
  (C/'configs'/f'{method}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 protocol=dict(id=C.name,methods=METHODS,seeds=SEEDS,benchmark=physical,hard_continuous_seconds=43200,soft_deadline_seconds=37800,checkpoint_selection='last successful native-target state; soft-deadline state separately time-limited, never selected by EIG',comparison='METHOD_SPECIFIC_NATIVE_SCHEDULES; NOT EQUAL_COMPUTE OR EQUAL_SAMPLES',targets=dict(fixed=dict(updates=50000,B=2000,L=2000,lr=.0001,gamma=.98),dad=dict(updates=50000,B=2000,L=2000,lr=.00005,gamma=.98),rl_sboed=dict(episodes=2000000,classification='PRACTICAL_BUDGET_LIMITED; not published convergence claim'),vsoed=dict(variant='G-I',cycles=10001,warmup=20000,new_per_cycle=1000,reason='Verified local native adapter hardcodes use_NFs=False; NF-I is not silently claimed'),scorebed=dict(score_updates=50000,policy_updates_per_restart=20000,restarts=5,batch=1024,reference='author pinned cc8e996e3ee3237ab4c9bd3ef80e2e4b4aa9cee4 config/experiment/location_finding.yaml; contradicts proposed190k/10k/P1',score_sampler='native configured random mixture, not old custom AR1 override')),source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),torch=torch.__version__,cuda=torch.version.cuda,precision='unchanged native float64 pathwise / native float32 RL,vsOED / JAX ScoreBED; TF32 disabled',exact_resume='NOT_ESTABLISHED: no automatic restart of interrupted native pipeline; preserve checkpoint and classify INCOMPLETE',final_bank='SEALED until explicit user MoE freeze',budgets='12h all-phase wall clock watchdog; native clock exclusions separately disclosed, never extends hard wall cap')
 put(C/'FORMAL_PROTOCOL.json',protocol)
 (C/'protocol_decisions.md').write_text('''# Prospective protocol decisions

Fixed/DAD schedules verified from https://github.com/ae-foster/dad/blob/main/README.md and location_finding.py; Adam betas .8/.998; schedule decay at update indices0,1000,... . Bounded common design map and source-prior initialization are explicit project adaptations; not bitwise author environment reproduction.

vsOED uses verified native **G-I**, not NF-I. The author README documents GMM incremental T30/n-incre30/discount.9. Local adapter explicitly use_NFs=False; introducing NF without validation is not warranted. 10001cycles plus20000 initial replay,1000fresh/cycle,5posterior/5critic, native learning settings retained. Native completion does not prove convergence.

ScoreBED authoritative pinned source cc8e996e3ee3237ab4c9bd3ef80e2e4b4aa9cee4 config/experiment/location_finding.yaml specifies **50000 score updates,20000 policy updates per restart,5 restarts**. It contradicts the proposed190000/10000/P1 setting. Use the verified pinned configuration including default random design mixture and batch1024. Native objectives/architectures/EMA remain unchanged. Independent diagnostic restart selection is charged in parent wall time; final common dev and sealed final banks never select restarts. Count simulator calls from executed ledger, not assumptions. Existing190k/P1 partial checkpoints are not reused.

RL: completed native2Mseed101 reused; native SAC/REDQ kernels/replay/critics verified in existing tests.2M is a prospective feasible project ceiling (~7h37m), not a paper convergence default. Replay/optimizer exact resume is unavailable; completed uninterrupted run reuse does not claim exact resume. Learning curve and remaining uncertainty disclosed.

No final-bank evaluation until the user freezes MoE. Existing four-hour/equal2M/partial runs remain separate unchanged evidence. This cohort is not equal-compute. Native-target completion and convergence are distinct.
''')
 # Revalidate surviving aligned historical models; do not copy/delete historical artifacts.
 prior=R/'scratch_runs/journal_location_T30_labpc_native4h_complete_20261101T194734Z/inventory.json';old=read(prior)['baselines'];two=R/'scratch_runs/journal_location_T30_2M_seed101_105_20261101T172522Z';rows=[]
 for seed in SEEDS:
  for method in METHODS:
   row=dict(method=method,seed=seed,decision='RERUN',status='QUEUED',checkpoint='',checkpoint_sha256='',reason='No fully completed protocol-aligned checkpoint verified',loader='native')
   if method=='random':row.update(decision='NO_TRAINING',status='EVAL_ONLY',reason='Untrained reference; independent policy RNG label')
   if seed==101 and method in ['dad','vsoed']:
    x=next(x for x in old if x['method']==method and x['seed']==seed);cp=pathlib.Path(x['checkpoint']);assert sha(cp)==x['checkpoint_sha256'];t=read(cp.with_name(method+'_training.json'))
    if method=='dad':
     effective=read(cp.with_name('dad_effective_settings.json'));assert t['actual_updates']==50000 and t['batch_size']==2000 and effective['contrasts']==2000
    else:assert t['profile']['updates']==10001 and t['profile']['n_incre']==30
    p=load_checkpoint(cp,make_benchmark(physical));assert all(torch.isfinite(v).all() for v in p.state_dict().values())
    row.update(decision='REUSE',status='REUSED_VERIFIED',checkpoint=str(cp),checkpoint_sha256=sha(cp),reason='Full native target/variant and source parity previously verified; strict checkpoint load rechecked; dev bank difference only requires reevaluation')
   if seed==101 and method=='rl_sboed':
    cp=two/'runs/seed101/rl_sboed/policy.pt';p=torch.load(cp,map_location='cpu',weights_only=False);t=read(cp.parent/'training.json');assert t['generated_training_trajectories']==2000000;assert t['gradient_updates']==1277888
    assert all(torch.isfinite(v).all() for v in p.state_dict().values());row.update(decision='REUSE',status='REUSED_VERIFIED',checkpoint=str(cp),checkpoint_sha256=sha(cp),loader='object',reason='Uninterrupted completed native2M target;7h37m; no exact-resume claim; PRACTICAL_BUDGET_LIMITED')
   rows.append(row)
 put(C/'jobs.json',rows);table(C/'reuse_matrix.csv',rows)
 (C/'journal_methods.tex').write_text('% Method-specific native schedules, not equal compute.\n\\begin{tabular}{lll}Method & Schedule & Caveat \\\\ Fixed & 50k,B2000,L2000 & bounded adaptation \\\\ DAD & 50k,B2000,L2000 & native finite target \\\\ RL & 2M episodes & practical budget limited \\\\ vsOED & G-I,10001 cycles & not NF-I \\\\ ScoreBED & 50k score+5x20k policy & pinned author configuration \\\\ \\end{tabular}\n')
 (C/'journal_dev_results.tex').write_text('% PENDING: no new common-development-bank results yet.\n')
 put(C/'status.json',dict(status='PREPARED_PENDING_TESTS',jobs=len(rows),reused=sum(x['decision']=='REUSE' for x in rows)))
 print(C,flush=True)
def entry_command(method,seed,out,smoke=False):
 return [str(PY),str(C/'source/native_entry.py'),'--config',str(C/'configs'/f'{method}.yaml'),'--journal','--methods',method,'--seed',str(seed),'--train-only','--output',str(out),'--source-check-manifest',str(C/'source_manifest.json')]+(['--smoke'] if smoke else [])
def freeze():
 files={str(p.relative_to(C)):sha(p) for d in ['source','configs'] for p in (C/d).glob('*') if p.is_file()};files.update({f:sha(C/f) for f in ['FORMAL_PROTOCOL.json','physical.json','bank_manifest.json','reuse_matrix.csv','protocol_decisions.md']});put(C/'frozen_files.json',files)
def tests():
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',PYTHONPATH=str(R))
 names=['test_dad_reference_budget.py','test_compute_budget.py','test_fixed_budget_selection.py','test_primary_budget.py','test_location_bounded.py','test_original_protocol.py','test_vsoed_paper_protocol.py','test_official_scorebed.py']
 with (C/'logs/cpu_tests.log').open('w') as f:rc=subprocess.call([str(PY),'-m','pytest','-q',*[str(R/'tools/tests'/n) for n in names]],env=env,stdout=f,stderr=subprocess.STDOUT)
 put(C/'readiness/cpu.json',dict(status='PASS' if rc==0 else 'FAIL',returncode=rc));return rc
def profile_fixed():
 import torch,yaml,numpy as np
 from src.research.benchmarks import make_benchmark
 from src.research.policies import make_policy
 from src.research.engine import rollout
 cfg=yaml.safe_load((C/'configs/fixed.yaml').read_text());s=cfg['research']['settings']|cfg['research']['journal_protocol']['method_settings']['fixed'];torch.set_num_threads(4);times=[]
 with torch.device('cuda'):
  b=make_benchmark(cfg);p=make_policy(b,30,'fixed',s['hidden'],s);opt=torch.optim.Adam(p.parameters(),lr=.0001,betas=(.8,.998))
  for k in range(4):
   torch.cuda.synchronize();a=time.monotonic();opt.zero_grad();r=rollout(b,p,30,2000,772020+k,2000);loss=-r['info'][:,-1].mean();loss.backward();assert all(torch.isfinite(x.grad).all() for x in p.parameters());opt.step();torch.cuda.synchronize();times.append(time.monotonic()-a)
 projected=max(times[1:])*50000*1.35+600;put(C/'readiness/fixed_production_profile.json',dict(seconds_per_update=times,projected_seconds=projected,passes=projected<37800,B=2000,L=2000,finite_gradients=True));assert projected<37800
def evaluate(row,bank_name='development'):
 import torch,numpy as np
 from src.research.journal_validation import CommonValidation
 from src.research.journal_checkpoints import load_checkpoint
 from src.research.benchmarks import make_benchmark
 from src.research.deepseek_sboed_experiment import RandomControl
 torch.set_num_threads(4);raw=read(C/'physical.json');bank=read(C/'bank_manifest.json')[bank_name];assert sha(bank['path'])==bank['file_sha256'];path=C/'evaluations'/f'{bank_name}_{row["method"]}_seed{row["seed"]}.json'
 if path.exists():return
 v=CommonValidation(raw,30,bank['seed'],2000,1000000,device='cuda');data=np.load(bank['path']);v.truth=torch.from_numpy(data['truth'].copy());v.noise=torch.from_numpy(data['noise'].copy());v.bank=torch.from_numpy(data['contrasts'].copy());v.identity=bank['draws_sha256']
 if row['method']=='random':p=RandomControl(raw,bank['seed']+10000+row['seed'])
 else:
  assert sha(row['checkpoint'])==row['checkpoint_sha256'];p=torch.load(row['checkpoint'],map_location='cpu',weights_only=False) if row.get('loader')=='object' else load_checkpoint(row['checkpoint'],make_benchmark(raw))
 values,meta=v(p);np.save(path.with_suffix('.npy'),values);se=float(values.std(ddof=1)/math.sqrt(2000));put(path,dict(method=row['method'],seed=row['seed'],status=row['status'],eig_mean=float(values.mean()),eig_mcse=se,ci95=[float(values.mean())-1.96*se,float(values.mean())+1.96*se],meta=meta,checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256']))
 results=[read(p) for p in sorted((C/'evaluations').glob(f'{bank_name}_*.json'))];table(C/f'{bank_name}_eig.csv',results,['method','seed','status','eig_mean','eig_mcse','ci95','checkpoint','checkpoint_sha256'])
 summaries=[];paired=[]
 for m in METHODS:
  x=[r for r in results if r['method']==m];a=np.array([r['eig_mean'] for r in x]);summaries.append(dict(method=m,evaluated_seeds=len(a),mean=float(a.mean()) if len(a) else None,across_seed_SD=float(a.std(ddof=1)) if len(a)>1 else None,mean_case_MCSE=float(np.mean([r['eig_mcse'] for r in x])) if x else None,interpretation='policy randomizations only' if m=='random' else 'independent training seeds'))
 for r in results:
  for ref in ['random','dad','rl_sboed','vsoed']:
   other=C/'evaluations'/f'{bank_name}_{ref}_seed{r["seed"]}.npy'
   if ref==r['method'] or not other.exists():continue
   d=np.load(C/'evaluations'/f'{bank_name}_{r["method"]}_seed{r["seed"]}.npy')-np.load(other);se=float(d.std(ddof=1)/math.sqrt(2000));paired.append(dict(method=r['method'],reference=ref,seed=r['seed'],difference=float(d.mean()),paired_MCSE=se,CI_low=float(d.mean())-1.96*se,CI_high=float(d.mean())+1.96*se))
 table(C/f'{bank_name}_seed_summary.csv',summaries);table(C/f'{bank_name}_paired.csv',paired)
 if bank_name=='development':
  lines=['% Actual common development EIG; partial seed counts explicit.','\\begin{tabular}{lrrr}Method & Seeds & Mean EIG & Across-seed SD \\\\']
  for x in summaries:lines.append(f"{x['method']} & {x['evaluated_seeds']} & {x['mean'] if x['mean'] is not None else '--'} & {x['across_seed_SD'] if x['across_seed_SD'] is not None else '--'} \\\\")
  lines.append('\\end{tabular}');(C/'journal_dev_results.tex').write_text('\n'.join(lines))
def manage():
 ctl=(C/'controller.lock').open('a+');fcntl.flock(ctl,fcntl.LOCK_EX|fcntl.LOCK_NB);held=None;jobs=read(C/'jobs.json');verify();assert read(C/'readiness/cpu.json')['status']=='PASS'
 while held is None:
  if (C/'CANCEL').exists():put(C/'status.json',dict(status='CANCELLED_BEFORE_START'));return
  try:held=locks()
  except BlockingIOError:put(C/'status.json',dict(status='WAITING_GPU',manager_pid=os.getpid()));time.sleep(20)
 try:
  # Production-size Fixed profile precedes any formal run.
  with (C/'logs/fixed_profile.log').open('w') as f:rc=subprocess.call([str(PY),str(__file__),'profile_fixed'],stdout=f,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError('Production Fixed profile failed: see logs/fixed_profile.log')
  # Independent per-method failure isolation. No retry of a partly executed attempt.
  for row in jobs:
   if (C/'CANCEL').exists():put(C/'status.json',dict(status='STOPPED_AT_BOUNDARY',manager_pid=os.getpid()));break
   verify();m,s=row['method'],row['seed']
   if row['decision'] not in ['REUSE','NO_TRAINING']:
    out=C/'runs'/f'seed{s}'/m;out.parent.mkdir(exist_ok=True,parents=True)
    if out.exists():row.update(status='INCOMPLETE',reason='Existing attempt; no unproved resume or duplicate restart');put(C/'jobs.json',jobs);continue
    if m=='scorebed' and not (C/'readiness/scorebed_production_profile.json').exists():
     row.update(status='BLOCKED_NATIVE_PRODUCTION_PROFILE',reason='Author settings differ from old P1 schedule; full configuration/profile validation required before formal ScoreBED');put(C/'jobs.json',jobs);continue
    log=C/'logs'/f'{m}_seed{s}.log';start=time.monotonic()
    with log.open('w') as f:
     p=subprocess.Popen(entry_command(m,s,out),env=dict(os.environ,BOED_TRAINING_START_MONOTONIC=str(start),PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     row.update(status='RUNNING',pid=p.pid,log=str(log));put(C/'jobs.json',jobs)
     while p.poll() is None:
      put(C/'status.json',dict(status='RUNNING',manager_pid=os.getpid(),worker_pid=p.pid,method=m,seed=s,wall_seconds=time.monotonic()-start,log=str(log),output=str(out)))
      if time.monotonic()-start>43080:
       os.killpg(p.pid,signal.SIGTERM)
       try:p.wait(timeout=30)
       except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
       break
      time.sleep(10)
    cp=out/'models'/f'{m}.pt'
    if p.returncode==0 and cp.exists():
     t=read(cp.with_name(m+'_training.json'));complete=t.get('actual_updates')==50000 if m in ['fixed','dad'] else t.get('generated_training_trajectories',0)>=2000000 if m=='rl_sboed' else t.get('selected_update',t.get('actor_updates'))==10001 if m=='vsoed' else t.get('actual_score_updates')==50000 and t.get('actual_policy_updates')==100000
     row.update(status='PRACTICAL_BUDGET_LIMITED' if m=='rl_sboed' else 'COMPLETED_NATIVE' if complete else 'INCOMPLETE',checkpoint=str(cp),checkpoint_sha256=sha(cp),charged_whole_method_wall_seconds=time.monotonic()-start,native_schedule_complete=complete)
    else:row.update(status='FAILED',returncode=p.returncode,charged_whole_method_wall_seconds=time.monotonic()-start,no_automatic_restart=True)
    put(C/'jobs.json',jobs)
   if row['decision']=='NO_TRAINING' or row.get('checkpoint'):
    put(C/'status.json',dict(status='DEVELOPMENT_EVALUATION',manager_pid=os.getpid(),method=m,seed=s))
    with (C/'logs'/f'eval_{m}_seed{s}.log').open('w') as f:rc=subprocess.call([str(PY),str(__file__),'evaluate',m,str(s)],stdout=f,stderr=subprocess.STDOUT)
    row['development_evaluation']='COMPLETE' if rc==0 else 'FAILED';put(C/'jobs.json',jobs)
   table(C/'production_manifest.csv',jobs,['method','seed','decision','status','pid','checkpoint','checkpoint_sha256','charged_whole_method_wall_seconds','native_schedule_complete','development_evaluation','reason'])
  put(C/'status.json',dict(status='QUEUE_FINISHED_WITH_LIMITATIONS' if any(x['status'] in ['FAILED','INCOMPLETE','BLOCKED_NATIVE_PRODUCTION_PROFILE'] for x in jobs) else 'TRAINING_AND_DEVELOPMENT_COMPLETE_FINAL_SEALED',manager_pid=os.getpid()))
 except BaseException as e:put(C/'status.json',dict(status='BLOCKED',manager_pid=os.getpid(),error=str(e),traceback=traceback.format_exc()));raise
 finally:
  if held:
   for f in held:f.close()
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='freeze':freeze()
 elif mode=='tests':sys.exit(tests())
 elif mode=='profile_fixed':profile_fixed()
 elif mode=='manage':manage()
 elif mode=='evaluate':evaluate(next(x for x in read(C/'jobs.json') if x['method']==sys.argv[2] and x['seed']==int(sys.argv[3])))
