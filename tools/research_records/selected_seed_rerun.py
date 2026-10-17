"""Three explicitly requested original-setting repeats; never replaces originals."""
import os,pathlib,json,time,hashlib,subprocess,signal,fcntl,csv,traceback,shutil,threading
import numpy as np
R=pathlib.Path(os.environ.get('MOE_PROJECT_ROOT',pathlib.Path(__file__).resolve().parents[2]));C=R/'experiments/10162026/10162026_selected_seed_original_rerun_v1';S=C/'source';OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget';TASKS=[('moe',103),('switchmoe',103),('scorebed',102)]
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)
def status(state,**kw):write(C/'status.json',dict(state=state,utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),**kw))
def guard():
 for rel,h in read(C/'source_manifest.json')['sha256'].items():
  if sha(S/rel)!=h:raise RuntimeError('Frozen repeat source changed: '+rel)
 for name,h in read(C/'campaign_seal.json').items():
  if sha(C/name)!=h:raise RuntimeError('Repeat protocol changed: '+name)
 if shutil.disk_usage(R).free<10*1024**3:raise RuntimeError('Disk safety floor10GiB')
def wait_device(method,seed):
 start=time.monotonic();clear=0
 while True:
  guard();lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();jobs=[x.strip()for x in lines if '-m src.online_cli'in x or '-m tools.stability_campaign'in x or '/tmp/resume_rerun_queue.py'in x or 'full_three_method_rerun/worker.py'in x or '-m tools.journal_campaign'in x];p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True);gpu=[x.strip()for x in p.stdout.splitlines()if x.strip()and x.split(',',1)[0].strip()!=str(os.getpid())];clear=clear+1 if not jobs and not gpu else 0
  if clear>=2:return
  status('waiting_for_free_GPU',method=method,seed=seed,other_jobs=jobs,gpu_jobs=gpu)
  if time.monotonic()-start>86400:raise RuntimeError('GPU wait cap1day')
  time.sleep(15)
def run(method,seed):
 out=S/f'experiments/10162026/10162026_selected_seed_rerun/{method}_seed{seed}';marker=C/f'{method}_{seed}_completion.json'
 if marker.exists():
  e=read(marker)
  if e['status']!='complete':raise RuntimeError('Previous failed attempt preserved; no automatic retry')
  assert sha(out/'models'/f'{method}.pt')==e['checkpoint_sha256'];return e
 if out.exists():raise RuntimeError('Partial output exists; refusing overwrite')
 wait_device(method,seed);out.parent.mkdir(parents=True,exist_ok=True);cmd=['bash','run.sh','--config',str(C/'location_original_settings.yaml'),'--journal','--seed',str(seed),'--eval-seeds','1001','--eig-only','--source-check-manifest',str(C/'source_manifest.json'),'--output',str(out)]+(['--methods','scorebed']if method=='scorebed'else['--variant',method]);write(C/f'{method}_{seed}_command.json',dict(command=cmd,cwd=str(S),training_episodes=1000000));status('training',method=method,seed=seed,run=str(out));entry=dict(method=method,seed=seed,run=str(out),status='running');start=time.monotonic();stop=threading.Event();samples=[]
 def watch():
  while not stop.is_set():
   p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True);samples.append(dict(seconds=time.monotonic()-start,processes=p.stdout.strip()));stop.wait(10)
 w=threading.Thread(target=watch,daemon=True);w.start()
 try:
  with (C/f'{method}_{seed}_launcher.log').open('w')as f:
   proc=subprocess.Popen(cmd,cwd=S,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=10800)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise RuntimeError('Wall cap exceeded; partial run retained')
  if rc:raise RuntimeError('Root launcher returned '+str(rc))
  tr=read(out/'models'/f'{method}_training.json');assert tr['generated_training_trajectories']==1000000 and tr['checkpoint_selection']=='final_predeclared_fixed_budget' and tr['selected_checkpoint']['episodes']==1000000
  if method!='scorebed':
   assert tr['effective_training_settings'].get('moe_router_minimum_gamma',.001)==.001
   import torch
   ck=torch.load(out/'models'/f'{method}.pt',map_location='cpu',weights_only=False);assert abs(ck['gamma']-.001)<1e-12
  rows=read(out/'eval'/f'{method}_seed1001.json');oldentry=next(x for x in read(OLD/'registry.json')if x['benchmark']=='location'and x['method']==method and x['seed']==seed);old=pathlib.Path(oldentry['run']);oldrows=read(old/'eval'/f'{oldentry["native_method"]}_seed1001.json');assert len(rows)==len(oldrows)==2000
  assert [[x['truth'],x['measurement_noise_standard']]for x in rows]==[[x['truth'],x['measurement_noise_standard']]for x in oldrows];assert sha(out/'eval/contrast_bank_seed1001.npy')==sha(old/'eval/contrast_bank_seed1001.npy')
  x=np.array([z['spce']for z in rows]);y=np.array([z['spce']for z in oldrows]);d=x-y;se=float(d.std(ddof=1)/len(d)**.5)
  entry.update(status='complete',original_eig=float(y.mean()),repeat_eig=float(x.mean()),paired_difference=float(d.mean()),paired_mc_se=se,paired_mc_ci_low=float(d.mean()-1.96*se),paired_mc_ci_high=float(d.mean()+1.96*se),primary_training_episodes=1000000,training_seconds=tr['training_seconds'],validation_seconds=tr['validation_seconds'],checkpoint_sha256=sha(out/'models'/f'{method}.pt'),original_table_replaced=False)
 except BaseException as exc:entry.update(status='failed',error=str(exc));raise
 finally:stop.set();w.join(timeout=12);entry.update(wall_seconds=time.monotonic()-start,gpu_samples=samples);write(marker,entry)
 return entry
def report(results):
 fields=['method','seed','status','original_eig','repeat_eig','paired_difference','paired_mc_ci_low','paired_mc_ci_high','primary_training_episodes','training_seconds','wall_seconds','run']
 with (C/'results.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
 write(C/'results.json',results);text=['# Requested original-setting repeats','', 'These are repeated trainings of selected development seeds. Original five-seed results are retained. No favorable-repeat selection or across-seed robustness claim.','', '|Method|Seed|Status|Original EIG|Repeat EIG|','|---|---:|---|---:|---:|']
 for x in results:text.append(f"|{x['method']}|{x['seed']}|{x['status']}|{x.get('original_eig','pending')}|{x.get('repeat_eig','pending')}|")
 (C/'report.md').write_text('\n'.join(text)+'\n')
def main():
 lock=(C/'device.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[]
 for m,s in TASKS:
  try:results.append(run(m,s))
  except Exception as exc:results.append(read(C/f'{m}_{s}_completion.json')if(C/f'{m}_{s}_completion.json').exists()else dict(method=m,seed=s,status='failed',error=str(exc)))
  report(results)
 status('complete'if all(x['status']=='complete'for x in results)else'finished_with_failures',requested=3,completed=sum(x['status']=='complete'for x in results),sir_deferred=True,automatic_additional_runs=False)
if __name__=='__main__':
 try:main()
 except BaseException as exc:status('stopped_requires_review',error=str(exc),traceback=traceback.format_exc());raise
