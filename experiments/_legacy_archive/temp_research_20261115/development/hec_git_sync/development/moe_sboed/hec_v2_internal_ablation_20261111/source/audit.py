"""One frozen HEC-v2 feature ablation audit; no optimizer or model writes."""
import os,sys,json,time,pathlib,hashlib,fcntl,threading,subprocess,resource,traceback
sys.dont_write_bytecode=True
HOME=pathlib.Path.home();R=HOME/'Documents/objective_power_grid_temp/scratch_runs';ROOT=pathlib.Path(__file__).resolve().parents[1];CFG=json.loads((ROOT/'CONFIG.json').read_text());P=R/CFG['source_run'];O=R/'moe_hec_v2_internal_ablation';os.environ['MOE_V2_RUN_ROOT']=str(P);sys.path.insert(0,str(P/'source'))
import numpy as np,torch
import evaluate_v2 as native
from native_common import original,sha,digest
from hec_policy import COST,mean_signal
from frozen_evaluator import contrast_scores
stats=native.stats

def put(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def status(s,reason,**kw):
 row=dict(status=s,reason=reason,time=time.time(),pid=os.getpid(),**kw);put(O/'STATUS.json',row);print(json.dumps(row),flush=True)
def gate(name):
 if name=='full' or name=='conventional_moe':return None
 if name=='uniform':return torch.full((30,4),.25,device='cuda')
 k=int(name[-1])-1;return torch.nn.functional.one_hot(torch.full((30,),k,device='cuda'),4).float()

def inputs():
 a=P/'arms/candidate/actor.pt';r=json.loads((P/'arms/candidate/RESULT.json').read_text());assert r['status']=='COMPLETE' and r['continuation_updates']==2000;assert sha(a)==r['actor_sha256'];old=json.loads((P/'ARTIFACT_SHA256.json').read_text())['files'];files={str(P/f):h for f,h in old.items() if f.startswith('source/')};files[str(a)]=r['actor_sha256']
 for f in ['evaluation/development_validation_bank.npz','evaluation/routing_uniforms.npy','evaluation/candidate_scores.npy','evaluation/EIG_RESULTS.json','ARTIFACT_SHA256.json','inputs/ordinary.pt']:
  p=P/f
  if p.exists():files[str(p)]=old[f] if f in old else sha(p)
 for f,h in files.items():assert sha(f)==h,f
 return files

def simulate(model,bank,name):
 return native.simulate(model,*[torch.tensor(bank[k],device='cuda') for k in ['truth','noise','u']],'ordinary' if name=='conventional_moe' else 'candidate',gates=gate(name))
def fuse(model,x,slots,g):
 pre=model.fusion[0](torch.cat([model.history(x),(g[...,None]*slots[...,:64]).flatten(1)],-1))+model.response_fusion((g[...,None]*slots[...,64:]).flatten(1));d=model.fusion[2](model.fusion[1](pre)).tanh();a0=model.base(x);return a0+4*d-a0*d.abs()

def validate():
 assert not (O/'PRE_EVALUATION_MANIFEST.json').exists();original.setup();lease=original.acquire();start=time.monotonic();protected=inputs();model,_=native.actor_load('candidate');before=digest(model.state_dict());assert model.dynamic and not model.stage_only;rng=np.random.default_rng(202611110601);bank=dict(truth=rng.standard_normal((4,4)),noise=rng.standard_normal((4,30)),u=rng.random((4,30)));contrasts=torch.tensor(rng.standard_normal((11,4)),device='cuda');outputs={};checks=[]
 with torch.no_grad():
  full=simulate(model,bank,'full');bench=type('B',(),dict(sigma=.5,calls=0))();score,_=contrast_scores(bench,full,contrasts,3,'cuda');literal=[]
  for i in range(4):
   th=torch.cat([full['truth'][i:i+1],contrasts]);cum=torch.zeros(12,device='cuda',dtype=torch.float64)
   for t in range(30):
    pred=mean_signal(th[None],full['actions'][t][i:i+1,None])[0,0];cum+=-.5*((full['observations'][t][i,0]-pred)/.5).square()-np.log(.5*np.sqrt(2*np.pi))
   literal.append(cum[0]-torch.logsumexp(cum,0)+np.log(12))
  torch.testing.assert_close(score[:,-1].to('cuda'),torch.stack(literal),rtol=1e-9,atol=1e-9);assert bench.calls==4*30*11
  for t in [0,12,29]:
   x=native.lib.native_x(full['actions'][:t],full['observations'][:t],4);a,info=model(x,collect=True);assert torch.equal(a,model(x));torch.testing.assert_close(a,fuse(model,x,info['slots'],info['gates']),rtol=1e-6,atol=1e-6)
   for name in ['expert1','expert2','expert3','expert4','uniform']:
    g=gate(name)[t:t+1].expand(4,-1);v,i=model(x,routing_override=g,collect=True);assert torch.equal(i['gates'],g);torch.testing.assert_close(v,fuse(model,x,i['slots'],g),rtol=1e-6,atol=1e-6)
    if name.startswith('expert'):
     k=int(name[-1])-1;changed=i['slots'].clone();mask=torch.arange(4,device='cuda')!=k;changed[:,mask]=torch.randn_like(changed[:,mask])*100;torch.testing.assert_close(v,fuse(model,x,changed,g),rtol=1e-6,atol=1e-6);checks.append(name+':both feature channels masked')
  for name in ['full','expert1','expert2','expert3','expert4','uniform']:
   a=simulate(model,bank,name);assert all(torch.isfinite(v).all() and bool((v.abs()<=4).all()) for v in a['actions']);outputs[name]=torch.stack(a['actions'],1)
  again=simulate(model,bank,'full');assert torch.equal(torch.stack(again['actions'],1),outputs['full']);assert digest(model.state_dict())==before;assert not any(p.requires_grad for p in model.parameters());compatible=True;reason=None
  try:
   conventional,_=native.actor_load('ordinary');v=simulate(conventional,bank,'conventional_moe');assert all(torch.isfinite(z).all() and bool((z.abs()<=4).all()) for z in v['actions'])
  except Exception as exc:compatible=False;reason=str(exc)
 for f,h in protected.items():assert sha(f)==h,f
 put(O/'evidence/VALIDATION.json',dict(status='PASS',seconds=time.monotonic()-start,architecture='four128dim feature packets;first64+response64;not complete designs',onehot_first_and_response_channels_verified=True,unselected_slots_cannot_affect_shared_fusion=True,unchanged_full_policy_bitwise=True,all6_feasible_at_all30steps=True,literal_positive_plus11negative_estimator_verified=True,primary_checks=checks,conventional_compatible=compatible,conventional_blocker=reason,model_unchanged=True,no_training=True,fixture_simulator_observations=4*30*(9 if compatible else 8)));print('HECV2_INTERNAL_ABLATIONS_VALIDATED',flush=True)

def monitor(stop,phase):
 before=time.monotonic();c0=time.process_time()
 while not stop.wait(CFG['telemetry_seconds']):
  try:
   t=time.monotonic();c=time.process_time();v=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',');rss=int(pathlib.Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE');row=dict(time=time.time(),policy=phase['name'],CPU_percent_one_core=100*(c-c0)/(t-before),RSS_bytes=rss,GPU_percent=float(v[0]),GPU_memory_MiB=float(v[1]),GPU_power_W=float(v[2]));before=t;c0=c
   with (O/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  except Exception as exc:
   with (O/'telemetry_errors.log').open('a') as f:f.write(str(exc)+'\n')

def run():
 guard=(O/'.run.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (O/'RUN_STARTED.json').exists(),'Duplicate evaluation prevented';manifest=json.loads((O/'PRE_EVALUATION_MANIFEST.json').read_text())
 for f,h in manifest['files'].items():assert sha(O/f)==h,f
 assert json.loads((O/'evidence/VALIDATION.json').read_text())['status']=='PASS';put(O/'RUN_STARTED.json',dict(pid=os.getpid(),time=time.time(),git_commit=manifest['git_commit']));original.setup()
 while True:
  try:lease=original.acquire();break
  except BlockingIOError:status('QUEUED','Waiting for idle LabPC GPU');time.sleep(30)
 status('RUNNING','Frozen seven-configuration evaluation');start=time.monotonic();deadline=start+CFG['evaluation_budget_seconds'];protected=inputs();put(O/'evidence/PROTECTED_INPUTS_BEFORE.json',protected);path=P/'evaluation/development_validation_bank.npz';bank=dict(np.load(path));bank['u']=np.load(P/'evaluation/routing_uniforms.npy');assert bank['truth'].shape==(2000,4) and bank['noise'].shape==(2000,30) and bank['contrasts'].shape==(1000000,4) and bank['u'].shape==(2000,30);assert sha(path)==json.loads((P/'evaluation/EIG_RESULTS.json').read_text())['bank_sha256'];neg=torch.tensor(bank['contrasts'],device='cuda');actor,_=native.actor_load('candidate');before=digest(actor.state_dict());result={};arrays={};stop=threading.Event();phase=dict(name='setup');thread=threading.Thread(target=monitor,args=(stop,phase),daemon=True);thread.start()
 try:
  compatible=json.loads((O/'evidence/VALIDATION.json').read_text())['conventional_compatible']
  for name in CFG['configurations']:
   if name=='conventional_moe' and not compatible:continue
   model=native.actor_load('ordinary')[0] if name=='conventional_moe' else actor;model_before=digest(model.state_dict());phase['name']=name;torch.cuda.reset_peak_memory_stats();COST.clear();t0=time.monotonic();c0=time.process_time();vals=[];curves=[];obs=0;batch_calls=0;bench=type('B',(),dict(sigma=.5,calls=0))()
   with torch.no_grad():
    for i in range(0,2000,16):
     if time.monotonic()>deadline:raise TimeoutError('Fixed evaluation budget')
     sl=slice(i,i+16);data=simulate(model,{k:bank[k][sl] for k in ['truth','noise','u']},name);v,_=contrast_scores(bench,data,neg,CFG['contrast_chunk'],'cuda');assert torch.isfinite(v).all();vals.extend(v[:,-1].numpy());curves.append(v.numpy());obs+=sum(y.numel() for y in data['observations']);batch_calls+=len(data['observations'])
     if (i//16)%40==0:status('RUNNING','Evaluating frozen configuration',configuration=name,cases_completed=min(2000,i+16))
    eval_seconds=time.monotonic()-t0;cpu_seconds=time.process_time()-c0;peak=torch.cuda.max_memory_allocated();queries=dict(COST);x=native.lib.native_x(data['actions'][:12],data['observations'][:12],len(data['truth']))[:1];g=None if gate(name) is None else gate(name)[12:13]
    for _ in range(5):model(x) if name=='conventional_moe' else model(x,routing_override=g)
    torch.cuda.synchronize();lat0=time.perf_counter()
    for _ in range(CFG['latency_repeats']):model(x) if name=='conventional_moe' else model(x,routing_override=g)
    torch.cuda.synchronize();latency=(time.perf_counter()-lat0)*1000/CFG['latency_repeats']
   values=np.asarray(vals);arrays[name]=values;np.save(O/f'{name}_scores.npy',values);np.save(O/f'{name}_curves.npy',np.concatenate(curves));result[name]=dict(**stats(values),evaluation_seconds=eval_seconds,process_CPU_seconds=cpu_seconds,GPU_peak_allocated_bytes=peak,process_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,simulator_episodes=2000,simulator_observations=obs,simulator_batch_calls=batch_calls,negative_observation_density_terms=bench.calls,positive_observation_density_terms=2000*30,model_queries=queries,inference_ms_per_experiment=latency);assert obs==60000 and bench.calls==60000000000;assert digest(model.state_dict())==model_before;put(O/f'{name}_receipt.json',result[name]);print('CONFIGURATION_COMPLETE',name,json.dumps(result[name]),flush=True)
   if name=='full':error=float(np.max(np.abs(values-np.load(P/'evaluation/candidate_scores.npy'))));assert error<=1e-8,('Original full policy reproduction failed',error)
   if name=='conventional_moe' and (P/'evaluation/ordinary_scores.npy').exists():assert np.max(np.abs(values-np.load(P/'evaluation/ordinary_scores.npy')))<=1e-8
  assert digest(actor.state_dict())==before;comparisons={name:stats(arrays['full']-arrays[name]) for name in ['expert1','expert2','expert3','expert4','uniform']};native.holm(comparisons,'Holm5_p');secondary=stats(arrays['full']-arrays['conventional_moe']) if 'conventional_moe' in arrays else None
  if secondary is not None:
   six={k:dict(v) for k,v in comparisons.items()};six['conventional_moe']=dict(secondary);native.holm(six,'Holm6_p');secondary['supplementary_Holm6_p']=six['conventional_moe']['Holm6_p']
  best=max(['expert1','expert2','expert3','expert4'],key=lambda k:result[k]['mean']);put(O/'RESULTS.json',dict(status='COMPLETE',methods=result,primary_comparisons=comparisons,conventional_comparison=secondary,best_observed_individual_expert=best,full_minus_best=comparisons[best],full_outperforms_all_four_corrected=all(comparisons[k]['mean']>0 and comparisons[k]['Holm5_p']<.05 for k in ['expert1','expert2','expert3','expert4']),learned_beats_uniform_corrected=comparisons['uniform']['mean']>0 and comparisons['uniform']['Holm5_p']<.05,bank_sha256=sha(path),actor_sha256=sha(P/'arms/candidate/actor.pt'),full_reproduction_max_error=error,total_seconds=time.monotonic()-start,finite_contrast_lower_bound=True,single_lineage=True,reused_development_bank=True))
 finally:stop.set();thread.join(timeout=8)
 for f,h in protected.items():assert sha(f)==h,f
 put(O/'evidence/PROTECTED_INPUTS_AFTER.json',protected);write_reports();status('COMPLETED','Frozen ablations and artifact preservation verified',report=str(O/'EIG_RESULTS.md'));put(O/'ARTIFACT_SHA256.json',{str(f.relative_to(O)):sha(f) for f in O.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ['ARTIFACT_SHA256.json','run.log']})

def write_reports():
 r=json.loads((O/'RESULTS.json').read_text());methods=r['methods'];tests=r['primary_comparisons'];best=r['best_observed_individual_expert'];fmt=lambda v:json.dumps(v,indent=2);block=lambda v:'```json\n'+fmt(v)+'\n```\n'
 table='|Configuration|EIG nats|MCSE|\n|---|---:|---:|\n'+''.join(f"|{k}|{v['mean']:.6f}|{v['MCSE']:.6f}|\n" for k,v in methods.items());(O/'EIG_RESULTS.md').write_text('# Frozen HEC-v2 internal ablation\n\n'+table+'\n2000 full trajectories,T30,1M negative contrasts plus positive once. Finite-contrast lower bound,not exact EIG. One training lineage;the original independent-from-training development bank is reused for this diagnostic. No final-test or retraining.\n\nBest observed individual expert:'+best+'\nFull policy reproduces its saved scores with maximum error:'+str(r['full_reproduction_max_error'])+'\n')
 (O/'PAIRED_COMPARISONS.md').write_text('# Paired full-minus-ablation comparisons\n\n'+block(tests)+'\nFive planned comparisons use Holm5;displayed95%tCIs are unadjusted. The best-observed-expert comparison belongs to this corrected family;its empirical winner label does not prove the population ranking.\n\nFull minus compatible conventional direct-head MoE(secondary;raw and supplementary Holm6 p):\n\n'+block(r['conventional_comparison']))
 (O/'EXPERT_SPECIALIZATION_ASSESSMENT.md').write_text('# Expert-pathway utility\n\n'+block(dict(best_observed_expert=best,full_minus_best=r['full_minus_best'],full_significantly_beats_all_four=r['full_outperforms_all_four_corrected']))+'\nExpert-only means one trained feature packet through the original shared head,with shared history and frozen SG reference retained. It is not a standalone expert design network. A full-policy gain over all one-hot controls would support collaboration within this checkpoint,not prove separately learned conditional competencies or causality of specialist training. One-hot weights change feature scales/distributions;negative and inconclusive results remain valid. No forced chronological roles.\n')
 (O/'ROUTING_ASSESSMENT.md').write_text('# Learned versus equal feature coordination\n\n'+block(dict(full_minus_uniform=tests['uniform'],corrected_positive=r['learned_beats_uniform_corrected']))+'\nUniform applies.25 to both channels of each feature packet at every step. It does not average physical designs or sample experts. The commander remains history dependent in every configuration;this isolates coordination weights,not commander removal. Non-significance is not equality. A gain would be a frozen-checkpoint effect;training-level claims need matched controls.\n')
 rows=[json.loads(x) for x in (O/'resources.jsonl').read_text().splitlines()] if (O/'resources.jsonl').exists() else [];resource_summary={}
 for k in methods:
  subset=[v for v in rows if v['policy']==k];resource_summary[k]={name:dict(sample_mean=float(np.mean([v[name] for v in subset])),sample_max=float(max(v[name] for v in subset))) for name in ['CPU_percent_one_core','RSS_bytes','GPU_percent','GPU_memory_MiB','GPU_power_W']} if subset else dict(status='NO_SAMPLES');resource_summary[k]['samples']=len(subset)
 (O/'COMPUTE_LEDGER.md').write_text('# Actual evaluation compute\n\n'+block(dict(total_seconds=r['total_seconds'],per_configuration=methods,sampled_resources=resource_summary))+'\nDevice telemetry sampled every2seconds;GPU memory samples and Torch allocated peaks are different measurements. CPU utilization is relative to one core;peak RAM uses Linux process high-water RSS. Per-policy wall/CPU clocks cover trajectory generation and contrast evaluation;single-case inference latency includes the shared frozen provider and all dense expert computations. No optimizer steps. No sparse-compute shortcut for disabled experts. Negative/positive observation-density terms are distinct from physical simulator observations and model-query counts.\n')
 (O/'ICML_IMPLICATIONS.md').write_text('# Evidence and limits\n\n'+block(dict(full_beats_every_individual=r['full_outperforms_all_four_corrected'],learned_beats_uniform=r['learned_beats_uniform_corrected'],full_minus_conventional=r['conventional_comparison']))+'\nThese tests answer frozen internal collaboration/coordination questions. They do not by themselves establish expert-specific history-predictable information competencies,new MoE methodology,robustness across training seeds,or an ICML-level contribution. A conventional-MoE difference is not compute matched and can reflect training/provider/capacity differences. The reused development bank is not an independent confirmatory final test. Preserve original HEC-v2;no further training or architecture search is launched.\n')
if __name__=='__main__':
 if '--validate' in sys.argv:validate()
 else:
  try:run()
  except Exception as exc:traceback.print_exc();status('BLOCKED',str(exc),error_type=type(exc).__name__)
