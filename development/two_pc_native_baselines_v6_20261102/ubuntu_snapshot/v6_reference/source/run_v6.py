import os,sys,pathlib,json,time,hashlib,subprocess,datetime,csv,math,signal,fcntl,traceback,io
R=pathlib.Path.home()/'Documents/objective_power_grid_temp';os.chdir(R);sys.path.insert(0,str(R));sys.path.insert(0,str(pathlib.Path(__file__).parent))
C=pathlib.Path(__file__).parent.parent;PY=pathlib.Path.home()/'miniconda3/envs/mocu_optimized/bin/python'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False));q.replace(p)
def read(p):return json.loads(pathlib.Path(p).read_text())
def load_canonical_bank(v):
 import numpy as np,torch
 path=C/'reference/development_bank.npz'
 assert sha(path)=='c3171762dc21926874b273adca01971b8d3833e30ae9355d9df3166417bf764e'
 data=np.load(path);v.truth=torch.from_numpy(data['truth'].copy());v.noise=torch.from_numpy(data['noise'].copy());v.bank=torch.from_numpy(data['contrasts'].copy())
 v.identity=hashlib.sha256(v.truth.numpy().tobytes()+v.noise.numpy().tobytes()+v.bank.numpy().tobytes()).hexdigest()
 assert v.identity=='a2cc2e6bea0f8b1dbaf7b7f30574b9bc5770729b53a758e5228ab851c5f99ff3'
 return v
def locks():
 held=[]
 try:
  for p in [R.parent/'objective_power_grid/.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock']:
   p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');held.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if busy:raise RuntimeError('WAITING_GPU: '+busy)
  return held
 except:
  for f in held:f.close()
  raise
def verify():
 for p,h in read(C/'source_manifest_sha256.json')['files'].items():
  file=C/p[5:] if p.startswith('SELF:') else R/p
  if sha(file)!=h:raise RuntimeError('Frozen source changed '+p)
def initialize():
 import torch,yaml,numpy as np
 from src.research.deepseek_sboed import DeepSeekPolicy
 from src.research.journal_validation import CommonValidation
 from v6_policy import V6Policy
 torch.set_num_threads(4);torch.manual_seed(101)
 assert sha(C/'reference/historical_original_v1.pt')=='f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7'
 historical=torch.load(C/'reference/historical_original_v1.pt',map_location='cpu',weights_only=False)
 histcfg=yaml.safe_load((C/'reference/historical_config.yaml').read_text());raw=histcfg['benchmark'];put(C/'physical.json',raw)
 p=DeepSeekPolicy();p.load_state_dict(historical['state_dict'],strict=True)
 torch.manual_seed(101);fresh=DeepSeekPolicy();torch.save(fresh.state_dict(),C/'fresh_shared_initial_state.pt')
 v=V6Policy();v.load_state_dict(fresh.state_dict(),strict=True)
 assert sum(x.numel() for x in v.parameters())==8434
 bank=load_canonical_bank(CommonValidation(raw,30,2026110117081001,2000,1000000))
 assert bank.identity=='a2cc2e6bea0f8b1dbaf7b7f30574b9bc5770729b53a758e5228ab851c5f99ff3'
 np.savez(C/'development_bank.npz',truth=bank.truth.numpy(),noise=bank.noise.numpy(),contrasts=bank.bank.numpy())
 protocol=dict(status='IMPLEMENTED_PENDING_TEST_GATE',training_seed=101,training_order=['v6','v1_matched'],updates=18000,primary_per_update=2000,primary_total=36000000,contrasts=2000,horizon=30,architecture=fresh.counts(),learning_rate=5e-5,adam_betas=[.8,.998],lr_decay=.98,original_lr_interval=1000,original_update_horizon=50000,common_lr_interval=360,common_lr_interpretation='Rescale original50k schedule to18k for BOTH arms: gamma.98 after first update then each360; common control adaptation, not uninterrupted historical V1.',gradient_clip=5,balance_factor=.001,exposure_penalty=0.,gumbel_initial_scale=.10,gumbel_plateau_fraction=.2,gumbel_end_fraction=.8,gumbel_schedule='0.10 until20%; cosine to0 at80%; zero for remainder; eval always0',gate_weights='original unperturbed softmax, gathered at noisy top3; not renormalized; hard indices, no STE; selected expert gradients plus smooth softmax/router gradients, no gradient through sampled membership',initial_state_sha256=sha(C/'fresh_shared_initial_state.pt'),routing_seed=7100101,rollout_seed_mapping='101 + successful_update*104729, same both arms; route RNG separate',time_cap=43200,safety_checkpoint_stop=41400,checkpoint_selection='last fully serialized18k-work checkpoint; never best EIG',evaluation=dict(seed=bank.seed,systems=2000,contrasts=1000000,bank_sha256=bank.identity,bank_file_sha256=sha(C/'development_bank.npz'),reserved_final='NOT COPIED, NOT USED;2026110117081002 remains reserved on LabPC'),historical_reference='original V1 f1107660,10.98906590nats; not contextual V1 b6800814,11.04404166nats',historical_not_matched=True,source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),software=dict(torch=torch.__version__,cuda=torch.version.cuda),tf32=False,matmul_precision='highest',native_simulator_changed=False,conditional_information_auxiliary=False)
 put(C/'protocol_frozen.json',protocol)
 files={str(p.relative_to(R)):sha(p) for p in (R/'src/research').rglob('*.py')}
 files.update({'SELF:'+str(p.relative_to(C)):sha(p) for p in (C/'source').glob('*.py')});files['SELF:physical.json']=sha(C/'physical.json');files['SELF:fresh_shared_initial_state.pt']=sha(C/'fresh_shared_initial_state.pt')
 put(C/'source_manifest_sha256.json',dict(files=files,head=protocol['source_head']))
 (C/'git_diff_summary.md').write_text('# Preserved working tree\n\n'+subprocess.check_output(['git','status','--short'],text=True)+'\nNo production source modified. New model/runner exist only in this scratch directory.\n')
 (C/'reference_v1_provenance.md').write_text('''# Verified V1 reference

Parent is original V1, source commit13681d7091189a4e68056e6a2864dc58f9d5700b. Current DeepSeekPolicy SHA85069fa88b27e62c82ab2fc326d36ac1aa573b01f5e4ceef2ffe0f4fd38e7573 matches that commit exactly. Original selected.pt SHA f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7 was read and SHA verified directly from LabPC legacy archive objective_power_grid_deepseekmoe_20261023/experiments/10232026/10232026_deepseekmoe_sboed/models/full/selected.pt. EIG10.98906589858419, MCSE.044794108684935746; bankseed1001,2000cases/1Mcontrasts; pairedSHA5545d4071aea78471408fae814302633f9ddead6da57942ffac7a28a72f07e0b;29,198updates/58,396,000mainpaths,14398.16137chargedseconds on LabPC4090. Historical one-unit discarded-readiness credit is separate from main-path exposure. Full original config and score JSON retained in reference/. Selected weights strictly load original8434-parameter class. They are used ONLY for architecture diagnostics, never new-arm initialization.

Separate contextual V1 source042915bf: selected SHA b680081479f1ca0227f250265642cea0452ce57e110d26220746d7966b50edfd was independently read/hash verified on LabPC; bank1001 pairedSHA identical; EIG11.044041656598598±.04281140219464484;26335updates/52,670,000mainpaths,14397.151256chargedseconds LabPC4090. It uses context-gradient corrections and is NOT the V6 parent or fresh matched control. Both historical references have different exposure and bank from this cohort. Native vsOED12.48085 is historical context only, not a paired result.\n''')
 (C/'v6_method.md').write_text('# V6 routing exploration\n\nOnly difference: training-time seeded Gumbel perturbations of original router logits choose top3. Gather original softmax weights at those indices; shared expert and fusion/head unchanged. Perturbations anneal to zero and eval always uses deterministic original routing. No extra parameters, information model, teacher, posterior, actor-critic, online search, or conditional-information loss. Existing V1 load balance.001 retained identically. Hard membership is not differentiated (native top-k approximation); gradients through selected expert features and full-softmax gathered gate weights remain. Exposure-control penalty OFF. Code audit found no stochastic route exploration in original V1.\n')
 for name in ['specialization_report.md','comparison_table.md','status_and_next_commands.md']:(C/name).write_text('# '+name+'\n\nPENDING. Useful specialization NOT VERIFIED. No new EIG yet.\n')
 put(C/'paired_eig_development.json',dict(status='PENDING',historical_not_matched=True))
 print('INITIALIZED',C,flush=True)
def model(arm,device='cpu'):
 import torch
 from src.research.deepseek_sboed import DeepSeekPolicy
 from v6_policy import V6Policy
 p=(V6Policy() if arm=='v6' else DeepSeekPolicy()).to(device)
 p.load_state_dict(torch.load(C/'fresh_shared_initial_state.pt',map_location=device,weights_only=False),strict=True);return p
def save(p,opt,sched,path,update,start,elapsed_credit=0.):
 import torch,numpy as np
 state=dict(method='moe_sboed_v6' if hasattr(p,'exploration_scale') else 'matched_original_v1',state_dict=p.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),update=update,primary=update*2000,seed=101,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],numpy_rng=np.random.get_state(),route_rng=p.route_generator.get_state() if getattr(p,'route_generator',None) is not None else None,exploration_scale=getattr(p,'exploration_scale',0.),protocol_sha256=sha(C/'protocol_frozen.json'),charged_seconds_before_serialization=time.monotonic()-start+elapsed_credit)
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
 return dict(path=str(path),sha256=sha(path),update=update,primary=update*2000,charged_seconds=time.monotonic()-start+elapsed_credit)
def gate():
 import torch,numpy as np
 from src.research.engine import rollout
 from src.research.benchmarks import make_benchmark
 from v6_policy import exploration
 torch.set_num_threads(4);raw=read(C/'physical.json');records={};a=model('v1_matched').eval();b=model('v6').eval()
 r1=rollout(make_benchmark(raw),a,30,4,91821120,7);r2=rollout(make_benchmark(raw),b,30,4,91821120,7)
 assert all(torch.equal(x,y) for x,y in zip(r1['actions'],r2['actions']));assert torch.equal(r1['info'],r2['info'])
 assert set(a.state_dict())==set(b.state_dict());assert all(x.abs().max()<=4 for x in r2['actions'])
 b.train();b.exploration_scale=.1;r=rollout(make_benchmark(raw),b,30,32,91821121,7);loss=-r['info'][:,-1].mean()+b.balance_loss();loss.backward()
 coverage=[sum(float(p.grad.abs().sum()) for p in e.parameters() if p.grad is not None)>0 for e in b.routed]
 assert all(coverage);assert sum(float(p.grad.abs().sum()) for p in b.router.parameters())>0
 assert sum(float(p.grad.abs().sum()) for p in b.shared.parameters())>0
 assert all(torch.isfinite(x.grad).all() for x in b.parameters() if x.grad is not None)
 assert all(ids.shape[-1]==3 for _,ids in b._gate_cache)
 b.route_generator=None;replay1=rollout(make_benchmark(raw),b,30,4,91821123,7)
 b.route_generator=None;replay2=rollout(make_benchmark(raw),b,30,4,91821123,7)
 assert torch.equal(replay1['info'],replay2['info'])
 import io
 buf=io.BytesIO();torch.save(b,buf);buf.seek(0);copy=torch.load(buf,weights_only=False).eval();b.eval()
 z1=rollout(make_benchmark(raw),b,30,4,91821122,7);z2=rollout(make_benchmark(raw),copy,30,4,91821122,7);assert torch.equal(z1['info'],z2['info'])
 assert exploration(1)==.1 and exploration(14401)==0 and exploration(18000)==0
 records.update(status='PASS',V6_OFF_bitwise_V1_actions_and_terminal_spce=True,all7_gradient_coverage=coverage,router_and_shared_gradients='finite and nonzero',exact_top3=True,strict_state_dict_compatible=True,seeded_roundtrip=True,feasibility_T30=True,historical_weights_not_training_initialization=True)
 put(C/'unit_test_results.json',records);(C/'unit_test_output.txt').write_text(json.dumps(records,indent=2))
 held=locks();cost={}
 try:
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
  for arm in ['v1_matched','v6']:
   t=time.monotonic();p=model(arm,'cuda').train();opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));sched=torch.optim.lr_scheduler.ExponentialLR(opt,.98);times=[];torch.cuda.reset_peak_memory_stats()
   for k in range(1,5):
    torch.cuda.synchronize();u=time.monotonic()
    if arm=='v6':p.exploration_scale=.1
    opt.zero_grad()
    with torch.device('cuda'):rr=rollout(make_benchmark(raw),p,30,2000,91821130+k*104729,2000)
    info=rr['info'][:,-1].mean();loss=-info+p.balance_loss();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-u)
   with torch.no_grad():
    logits=p.router(p.last_features);std=float(logits.std());counts=p.routing_statistics()
   io_start=time.monotonic();save(p,opt,sched,C/'readiness'/f'{arm}_discarded_smoke.pt',4,t);io_cost=time.monotonic()-io_start
   estimate=max(times[1:])*18000*1.5+600+io_cost*40
   cost[arm]=dict(unit_seconds=times,checkpoint_seconds=io_cost,conservative_estimate_seconds=estimate,passes=estimate<41400,peak_gpu_bytes=torch.cuda.max_memory_allocated(),router_logit_std=std,gumbel_scale=.1 if arm=='v6' else 0.,noise_to_logit_std=.1/std if std else None,routing=counts,profile_primary=8000,profile_updates=4,charged_profile_seconds=time.monotonic()-t)
   del p,opt,rr,loss,info,sched;torch.cuda.empty_cache()
  assert all(x['passes'] for x in cost.values()),'18k infeasible: do not run unmatched arms'
 finally:
  for f in held:f.close()
 put(C/'throughput_feasibility.json',cost);(C/'cuda_smoke_output.txt').write_text(json.dumps(cost,indent=2))
 print('GATE_PASS',cost,flush=True)
def train(arm):
 verify();import torch,numpy as np
 from src.research.engine import rollout
 from src.research.benchmarks import make_benchmark
 from v6_policy import exploration
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 out=C/('training_v6_seed101' if arm=='v6' else 'training_v1_matched_seed101');out.mkdir(exist_ok=False);(out/'checkpoints').mkdir();start=time.monotonic();torch.manual_seed(101);np.random.seed(101)
 protocol=read(C/'protocol_frozen.json');raw=read(C/'physical.json');put(out/'config.json',protocol);p=model(arm,'cuda').train();opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));sched=torch.optim.lr_scheduler.ExponentialLR(opt,.98);update=0;checkpoint=None
 try:
  for k in range(1,18001):
   if time.monotonic()-start>=41400:break
   scale=exploration(k) if arm=='v6' else 0.
   if arm=='v6':p.exploration_scale=scale
   opt.zero_grad()
   with torch.device('cuda'):
    r=rollout(make_benchmark(raw),p,30,2000,101+k*104729,2000);info=r['info'][:,-1].mean();loss=-info+p.balance_loss()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite terminal full-horizon sPCE')
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True);opt.step()
   update=k
   if (k-1)%360==0:sched.step()
   if k==1:
    put(out/'initial_draw_receipt.json',dict(initial_weights_sha256=sha(C/'fresh_shared_initial_state.pt'),prior_truth_sample_sha256=hashlib.sha256(r['truth'][:8].detach().cpu().numpy().tobytes()).hexdigest(),contrast_sample_sha256=hashlib.sha256(r['particles'][:2,:8].detach().cpu().numpy().tobytes()).hexdigest(),rollout_seed=101+104729))
   if k==1 or k%100==0:
    torch.cuda.synchronize();row=dict(status='RUNNING',arm=arm,pid=os.getpid(),update=k,primary=k*2000,contrast_particle_paths=k*2000*2000,simulator_parameter_stages=k*2000*2001*30,conditional_likelihood_terms=k*2000*2001*30,spce=float(info.detach()),finite_spce_bound=math.log(2001),loss=float(loss.detach()),gradient_norm=float(grad),exploration_scale=scale,learning_rate=opt.param_groups[0]['lr'],charged_seconds=time.monotonic()-start,routing=p.routing_statistics(),expert_gradient_norms=[float(torch.stack([x.grad.norm() for x in e.parameters() if x.grad is not None]).norm()) if any(x.grad is not None for x in e.parameters()) else 0. for e in p.routed])
    row['stage_expert_usage']=[torch.nn.functional.one_hot(ids,7).sum(1).double().mean(0).cpu().tolist() for _,ids in p._gate_cache]
    put(out/'status.json',row)
    with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    with (out/'compute_accounting.csv').open('a') as f:
     w=csv.writer(f)
     if k==1:w.writerow(['arm','successful_policy_updates','new_primary_trajectories','contrast_particle_paths','simulator_parameter_stages','conditional_likelihood_terms','charged_seconds','gumbel_scale'])
     w.writerow([arm,k,k*2000,k*2000*2000,k*2000*2001*30,k*2000*2001*30,row['charged_seconds'],scale])
   if k%1000==0 or k==18000:checkpoint=save(p,opt,sched,out/'checkpoints'/f'update{k}.pt',k,start)
  checkpoint=save(p,opt,sched,out/'checkpoints'/f'update{update}_final.pt',update,start)
  complete=update==18000 and checkpoint['charged_seconds']<43200
  result=dict(status='COMPLETE' if complete else 'INCOMPLETE',arm=arm,update=update,primary=update*2000,attempted_updates=update,skipped_updates=0,charged_seconds=checkpoint['charged_seconds'],checkpoint=checkpoint,peak_gpu_bytes=torch.cuda.max_memory_allocated(),cpu_process_seconds=time.process_time(),stop_reason='EXACT_WORK_TARGET_COMPLETE' if complete else 'SAFETY_TIME_CAP',training_simulator_parameter_stages=update*2000*2001*30,training_conditional_likelihood_terms=update*2000*2001*30,training_policy_updates=update,extra_counterfactual_simulations=0,gumbel_route_draws=min(update,14400)*30*2000*7 if arm=='v6' else 0)
  put(out/'result.json',result);put(out/'status.json',result)
  if not complete:sys.exit(3)
 except BaseException as e:
  if not (out/'result.json').exists():
   if update:checkpoint=save(p,opt,sched,out/'checkpoints'/f'update{update}_failure.pt',update,start)
   put(out/'failure.json',dict(status='INCOMPLETE',error=str(e),traceback=traceback.format_exc(),successful_updates=update,primary=update*2000,checkpoint=checkpoint))
  raise
def eval_arm(arm,mode='learned',swap_stage=None,swap_expert=None):
 import torch,numpy as np
 from src.research.journal_validation import CommonValidation
 from v6_policy import V6Policy
 raw=read(C/'physical.json');protocol=read(C/'protocol_frozen.json');out=C/('training_v6_seed101' if arm=='v6' else 'training_v1_matched_seed101');result=read(out/'result.json');assert result['status']=='COMPLETE'
 checkpoint=result['checkpoint'];assert sha(checkpoint['path'])==checkpoint['sha256'];state=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
 p=V6Policy();p.load_state_dict(state['state_dict'],strict=True);p.eval();p.diagnostic_mode=mode if mode!='learned' else None;p.swap_stage=swap_stage;p.swap_expert=swap_expert
 v=load_canonical_bank(CommonValidation(raw,30,protocol['evaluation']['seed'],2000,1000000,device='cuda'));assert v.identity==protocol['evaluation']['bank_sha256'];torch.set_num_threads(4)
 values,meta=v(p);label=f'{arm}_{mode}'+(f'_swap{swap_stage}_{swap_expert}' if swap_stage is not None else '')
 np.save(C/'evaluations'/f'{label}.npy',values);put(C/'evaluations'/f'{label}.json',dict(mean=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values))),meta=meta,checkpoint_sha256=checkpoint['sha256']))
def evaluate():
 verify();import numpy as np
 for arm in ['v6','v1_matched']:eval_arm(arm)
 a=np.load(C/'evaluations/v6_learned.npy');b=np.load(C/'evaluations/v1_matched_learned.npy');d=a-b;se=float(d.std(ddof=1)/math.sqrt(len(d)));r=dict(status='COMPLETE',V6=float(a.mean()),V1_matched=float(b.mean()),paired_difference=float(d.mean()),paired_MC_SE=se,CI95=[float(d.mean())-1.96*se,float(d.mean())+1.96*se],training_seed=101,independent_training_seed_replication=False,historical_scores_not_paired=True)
 put(C/'paired_eig_development.json',r);(C/'comparison_table.md').write_text('# Matched development comparison\n\n'+json.dumps(r,indent=2)+'\nHistorical10.98907/11.04404 and vsOED12.48085 have different bank/exposure; not matched.\n')
 # Offline controls preserve top3 count and unrenormalized native gate weights.
 diagnostic=[]
 for mode in ['fixed','stage_only']:
  eval_arm('v6',mode);x=np.load(C/'evaluations'/f'v6_{mode}.npy');delta=a-x;diagnostic.append(dict(control=mode,learned_minus_control=float(delta.mean()),paired_SE=float(delta.std(ddof=1)/math.sqrt(len(delta)))))
 for stage in [10,20]:
  for expert in range(7):
   eval_arm('v6','learned',stage,expert);x=np.load(C/'evaluations'/f'v6_learned_swap{stage}_{expert}.npy');delta=x-a
   diagnostic.append(dict(stage=stage,insert_expert=expert,swap_minus_learned=float(delta.mean()),paired_SE=float(delta.std(ddof=1)/math.sqrt(len(delta))),first_half_mean=float(delta[:1000].mean()),second_half_mean=float(delta[1000:].mean())))
 put(C/'specialization_diagnostics.json',diagnostic)
 (C/'specialization_report.md').write_text('# Specialization\n\nNOT VERIFIED automatically. Offline fixed/stage-only routing and single-decision expert insertions use identical truth/noise/contrasts. Swaps keep two original experts and replace one only when inserted expert absent, then resume current continuation policy. These population-average frozen interventions can shift distribution; they do not prove precise conditional causal competence. No claim of SPECIALIZED without independent context-effect replication and multiplicity-controlled analysis. Shuffled-routing control remains NOT COMPLETED.\n\n'+json.dumps(diagnostic,indent=2))
def supervise():
 held=locks();control=(C/'controller.lock').open('a+');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  verify();assert all(x['passes'] for x in read(C/'throughput_feasibility.json').values());put(C/'status.json',dict(status='RUNNING',supervisor_pid=os.getpid(),arm='v6'))
  for arm in ['v6','v1_matched']:
   with (C/'logs'/f'{arm}_seed101.log').open('a') as f:
    p=subprocess.Popen([str(PY),str(__file__),'train',arm],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);start=time.monotonic();put(C/'status.json',dict(status='RUNNING',supervisor_pid=os.getpid(),worker_pid=p.pid,arm=arm,log=str(C/'logs'/f'{arm}_seed101.log')))
    while p.poll() is None:
     if time.monotonic()-start>42000:
      os.killpg(p.pid,signal.SIGTERM)
      try:p.wait(timeout=30)
      except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
      break
     time.sleep(10)
   if p.returncode!=0:put(C/'status.json',dict(status='INCOMPLETE',arm=arm,returncode=p.returncode,no_automatic_restart=True));return
   with (C/'compute_accounting.csv').open('a') as f:
    result=read(C/('training_v6_seed101' if arm=='v6' else 'training_v1_matched_seed101')/'result.json');w=csv.writer(f)
    if arm=='v6':w.writerow(['arm','successful_updates','new_primary','charged_seconds','simulator_parameter_stages','likelihood_terms','checkpoint_sha256'])
    w.writerow([arm,result['update'],result['primary'],result['charged_seconds'],result['training_simulator_parameter_stages'],result['training_conditional_likelihood_terms'],result['checkpoint']['sha256']])
  with (C/'logs/evaluation.log').open('a') as f:
   p=subprocess.Popen([str(PY),str(__file__),'evaluate'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);put(C/'status.json',dict(status='EVALUATING',supervisor_pid=os.getpid(),worker_pid=p.pid));rc=p.wait()
  put(C/'status.json',dict(status='COMPLETE' if rc==0 else 'EVALUATION_FAILED',returncode=rc))
 finally:
  for f in held:f.close()
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='initialize':initialize()
 elif mode=='gate':gate()
 elif mode=='train':train(sys.argv[2])
 elif mode=='supervise':supervise()
 elif mode=='evaluate':evaluate()
