"""One no-training paired diagnostic; wait outside the sealed upstream tree."""
import os,sys,json,time,pathlib,hashlib,threading,subprocess,resource,fcntl,traceback
sys.dont_write_bytecode=True
Q=pathlib.Path(__file__).resolve().parents[1];CFG=json.loads((Q/'CONFIG.json').read_text());R=pathlib.Path.home()/'Documents/objective_power_grid_temp/scratch_runs';O=R/CFG['upstream_run'];D=O/CFG['result_subdir'];OUTPUT_OWNED=False
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else None
def put(p,v):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def status(state,reason,**extra):
 row=dict(status=state,reason=reason,time=time.time(),pid=os.getpid(),upstream=str(O),results=str(D),**extra);put(Q/'STATUS.json',row)
 if OUTPUT_OWNED:put(D/'STATUS.json',row)
 print(json.dumps(row),flush=True)
def alive(pid,expected):
 try:return expected in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
 except FileNotFoundError:return False

def imports():
 os.environ['MOE_V3_RUN_ROOT']=str(O);sys.path.insert(0,str(O/'source'))
 import torch,numpy as np,evaluate as native
 from common import original,stats,holm,digest
 from frozen_evaluator import contrast_scores
 return torch,np,native,original,stats,holm,digest,contrast_scores

def test():
 torch,np,native,original,stats,holm,digest,scorer=imports()
 from v3_policy import V3Actor
 class Stub:
  sigma=.025
  def parts(self,x,override=None):
   mu=x.new_tensor([[-1.2,-.7],[-.2,.6],[.8,-.3],[1.3,1.1]])[None].expand(len(x),-1,-1);g=x.new_tensor([.97,.01,.01,.01])[None].expand(len(x),-1);return g,mu,dict(proposals=4*mu.tanh())
 s=Stub();x=torch.zeros(1000,117);u=(torch.arange(1000)+.5)/1000;eps=torch.zeros(1000,2);g=torch.full((4,),.25);actions,info=V3Actor.draw(s,x,u,eps,gates=g);expected=(u[:,None]>g.cumsum(-1)).sum(-1)
 assert torch.equal(info['selected'],expected);assert torch.equal(torch.bincount(expected,minlength=4),torch.tensor([250]*4));assert torch.equal(info['gates'],torch.full((1000,4),.25));selected=info['proposals'][torch.arange(1000),expected];assert torch.equal(actions,selected);average=info['proposals'].mean(1);assert bool(((actions-average).abs().sum(-1)>0).all());assert not torch.cuda.is_initialized(),'CPU fixture must not initialize GPU while training'
 result=dict(status='PASS',actual_V3_draw_used=True,uniform_probabilities_exact=True,categorical_counts=[250]*4,selected_complete_design_not_coordinates_average=True,no_GPU_context=True,no_training=True);put(Q/'CPU_VALIDATION.json',result);print(json.dumps(result))

def telemetry(stop,phase):
 prev=time.monotonic();cpu0=time.process_time()
 while not stop.wait(CFG['telemetry_seconds']):
  try:
   now=time.monotonic();cpu=time.process_time();values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',');rss=int(pathlib.Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE');row=dict(time=time.time(),phase=phase['name'],CPU_percent_one_core=100*(cpu-cpu0)/(now-prev),RSS_bytes=rss,GPU_percent=float(values[0]),GPU_memory_MiB=float(values[1]),GPU_power_W=float(values[2]));prev=now;cpu0=cpu
   with (D/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  except Exception as exc:
   with (D/'telemetry_errors.log').open('a') as f:f.write(str(exc)+'\n')

def evaluate_pair():
 global OUTPUT_OWNED
 torch,np,native,original,stats,holm,digest,scorer=imports();original.setup();lease=None
 while lease is None:
  try:lease=original.acquire()
  except BlockingIOError:status('QUEUED','Upstream completed;waiting for idle GPU');time.sleep(30)
 assert not D.exists(),'Result directory already exists;never duplicate or overwrite';D.mkdir();OUTPUT_OWNED=True;status('RUNNING','Frozen paired evaluation acquired GPU');start=time.monotonic();deadline=start+CFG['evaluation_budget_seconds'];bank_path=O/'evaluation/development_validation_bank.npz';model_path=O/'arms/candidate/actor.pt';manifest=json.loads((O/'PRETRAIN_MANIFEST.json').read_text());protected={str(O/p):h for p,h in manifest['files'].items() if p.startswith('source/')};receipt=read(O/'arms/candidate/RESULT.json');assert receipt['status']=='COMPLETE' and receipt['updates']==2000;protected[str(model_path)]=receipt['actor_sha256']
 for p in [bank_path,O/'evaluation/candidate_scores.npy',O/'evaluation/EIG_RESULTS.json',O/'ARTIFACT_SHA256.json',O/'evidence/EXPERIMENT_COMPLETION.json']:
  protected[str(p)]=sha(p)
 for p,h in protected.items():assert sha(p)==h,p
 put(D/'PROTECTED_INPUTS_BEFORE.json',protected);bank=np.load(bank_path);assert bank['truth'].shape==(2000,4) and bank['noise'].shape==(2000,30) and bank['u'].shape==(2000,30) and bank['eps'].shape==(2000,30,2) and bank['contrasts'].shape==(1000000,4);old=read(O/'evaluation/EIG_RESULTS.json');assert protected[str(bank_path)]==old['new_bank_sha256'];model=native.load_model('candidate');assert not any(p.requires_grad for p in model.parameters());before=digest(model.state_dict());neg=torch.tensor(bank['contrasts'],device='cuda');result={};arrays={};stop=threading.Event();phase=dict(name='setup');thread=threading.Thread(target=telemetry,args=(stop,phase),daemon=True);thread.start()
 try:
  for name in ['learned','uniform']:
   phase['name']=name;torch.cuda.reset_peak_memory_stats();t0=time.monotonic();cpu0=time.process_time();counts=np.zeros((30,4),dtype=np.int64);scores=[];curves=[];bench=type('B',(),dict(sigma=.5,calls=0))();gates=torch.full((30,4),.25,device='cuda') if name=='uniform' else None
   with torch.no_grad():
    for i in range(0,2000,16):
     if time.monotonic()>deadline:raise TimeoutError('Fixed paired-evaluation budget')
     sl=slice(i,i+16);data=native.rollout(model,*[torch.tensor(bank[k][sl],device='cuda') for k in ['truth','noise','u','eps']],'candidate',collect=True,gates=gates)
     if name=='uniform':assert torch.equal(data['gates'],torch.full_like(data['gates'],.25))
     selected=data['selected'].cpu().numpy()
     if name=='uniform':assert np.array_equal(selected,(bank['u'][sl,:,None]>np.cumsum(np.full(4,.25))).sum(-1))
     for t in range(30):counts[t]+=np.bincount(selected[:,t],minlength=4)
     v,_=scorer(bench,data,neg,CFG['contrast_chunk'],'cuda');assert torch.isfinite(v).all();scores.extend(v[:,-1].numpy());curves.append(v.numpy())
     if (i//16)%25==0:status('RUNNING',f'{name} policy evaluation',policy=name,cases_completed=min(i+16,2000))
   array=np.asarray(scores);arrays[name]=array;np.save(D/f'{name}_scores.npy',array);np.save(D/f'{name}_curves.npy',np.concatenate(curves));np.save(D/f'{name}_selection_counts.npy',counts);total=counts.sum(0);assert total.sum()==60000
   result[name]=dict(**stats(array),evaluation_seconds=time.monotonic()-t0,process_CPU_seconds=time.process_time()-cpu0,selection_counts=total.tolist(),selection_frequencies=(total/60000).tolist(),selection_frequencies_by_stage=(counts/2000).tolist(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),process_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,simulator_episodes=2000,simulator_observations=60000,negative_density_terms=bench.calls);put(D/f'{name}_receipt.json',result[name]);assert digest(model.state_dict())==before
  reference=np.load(O/'evaluation/candidate_scores.npy');error=float(np.max(np.abs(arrays['learned']-reference)));assert error<=1e-8,('Learned same-bank reproduction mismatch',error);paired=stats(arrays['learned']-arrays['uniform']);family={f'original_{k}':dict(v) for k,v in old['primary'].items()};assert len(family)==2;family['learned_minus_uniform']=dict(paired);holm(family);result['paired_learned_minus_uniform']=dict(**paired,Holm3_p=family['learned_minus_uniform']['Holm_p']);result['combined_supplementary_Holm3']=family;result['provenance']=dict(actor_sha256=protected[str(model_path)],bank_sha256=protected[str(bank_path)],same_frozen_actor=True,training=False,cases=2000,T=30,negative_contrasts=1000000,learned_reproduction_max_error=error,finite_contrast_lower_bound_not_exact_EIG=True,reused_development_bank_not_final_test=True,single_training_lineage=True);result['total_evaluation_seconds']=time.monotonic()-start
 finally:stop.set();thread.join(timeout=8)
 rows=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()] if (D/'resources.jsonl').exists() else [];summary={}
 for p in ['learned','uniform']:
  subset=[x for x in rows if x['phase']==p];summary[p]={k:dict(sample_mean=float(np.mean([x[k] for x in subset])),sample_max=float(max(x[k] for x in subset))) for k in ['CPU_percent_one_core','RSS_bytes','GPU_percent','GPU_memory_MiB','GPU_power_W']} if subset else dict(status='NO_SAMPLES');summary[p]['samples']=len(subset)
 result['sampled_resources']=summary
 for p,h in protected.items():assert sha(p)==h,p
 put(D/'PROTECTED_INPUTS_AFTER.json',protected);put(D/'RESULTS.json',result);report='# Frozen V3:learned versus uniform routing\n\nSame final actor;2000 full trajectories,T30,1M prior negatives,identical truth/noise/routing uniforms/action normals. Finite-contrast EIG lower bound,not exact EIG.\n\n|Policy|EIG nats|MCSE|Evaluation seconds|\n|---|---:|---:|---:|\n'
 for p in ['learned','uniform']:report+=f"|{p}|{result[p]['mean']:.6f}|{result[p]['MCSE']:.6f}|{result[p]['evaluation_seconds']:.2f}|\n"
 report+='\nPaired learned-minus-uniform(ordinary95%CI;raw and supplementary Holm3 p):\n\n```json\n'+json.dumps(result['paired_learned_minus_uniform'],indent=2)+'\n```\n\nFull selection frequencies,per-stage counts and resource measurements are in RESULTS.json and resources.jsonl. Expert selection is sampled;coordinates are never averaged. Changed actions resimulate observations. No parameters or original outputs changed. Reused development bank and one trained lineage limit confirmatory claims;Holm3 augments the original two-primary family for this supplementary report without rewriting historical tests. Device utilization samples are not exact per-kernel attribution.\n';(D/'UNIFORM_ROUTER_EIG.md').write_text(report);status('COMPLETED','Paired evaluation and input preservation verified',results_json=str(D/'RESULTS.json'),report=str(D/'UNIFORM_ROUTER_EIG.md'))
 put(D/'ARTIFACT_SHA256.json',{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='ARTIFACT_SHA256.json'})

def main():
 lock=(Q/'.queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (Q/'RUN_STARTED.json').exists(),'One execution only';put(Q/'RUN_STARTED.json',dict(pid=os.getpid(),time=time.time()));source=read(Q/'SOURCE_LOCK.json');assert source and source['git_commit']
 for p,h in source['files'].items():assert sha(Q/p)==h,p
 assert read(Q/'CPU_VALIDATION.json')['status']=='PASS';status('QUEUED','Waiting for upstream V3 success and process exit');launch=read(O/'evidence/LAUNCH_RECEIPT.json')
 while True:
  completion=read(O/'evidence/EXPERIMENT_COMPLETION.json');active=bool(launch) and alive(launch['pid'],str(O/'source/driver.py'))
  if active:time.sleep(30);continue
  if not completion or completion['status']!='COMPLETE':raise RuntimeError('Upstream V3 did not complete successfully;no retraining')
  result=read(O/'arms/candidate/RESULT.json');assert result and result['status']=='COMPLETE' and result['updates']==2000,'No successful final V3 checkpoint';assert (O/'evaluation/development_validation_bank.npz').is_file();assert (O/'evidence/PRESERVATION_VERIFIED.json').is_file();break
 evaluate_pair()
if __name__=='__main__':
 if '--test' in sys.argv:test()
 else:
  try:main()
  except Exception as exc:
   traceback.print_exc();other=read(Q/'RUN_STARTED.json')
   if other and other['pid']!=os.getpid():print('DUPLICATE_PREVENTED',flush=True)
   else:status('BLOCKED',str(exc),error_type=type(exc).__name__)
