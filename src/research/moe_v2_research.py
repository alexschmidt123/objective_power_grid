"""Post-consolidation location MoE research. Root run.sh dispatches this module."""
import argparse,copy,fcntl,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch,yaml
from src.research.moe_v2_decision import DecisionMoE,matched_models
from src.research.journal_validation import CommonValidation
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout

ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT.with_name('objective_power_grid')/'experiments/_legacy_archive/ubuntu-pc/objective_power_grid_moe_v2_pilot_frozen_20261026/experiments/10262026/10262026_moe_sboed_v2_decision_pilot'
STAGES=[1,10,20,29]

def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(obj,indent=2,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(x):
 x=np.asarray(x);se=float(x.std(ddof=1)/len(x)**.5)
 return {'mean':float(x.mean()),'MC_SE':se,'N':len(x),'CI95':[float(x.mean()-1.96*se),float(x.mean()+1.96*se)]}
def source_manifest():
 files=[p for folder in ('src','tools','configs') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml') and '__pycache__' not in p.parts]
 return {'worktree_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'recovered_scientific_revision':'4b2bdfceca488f3b42be9d84c015dc8f9cf87a65','sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
def lock_gpu(out):
 p=Path.home()/'.cache/objective_power_grid/device_0.lock';p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+')
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 if active:raise RuntimeError('GPU has active compute processes: '+active)
 write(out/'status.json',{'state':'GPU_LOCK_ACQUIRED','pid':os.getpid(),'lock':str(p)})
 return f
def load_legacy():
 models=matched_models(201);record=json.loads((LEGACY/'metadata/final_checkpoint_freeze.json').read_text())
 for name,model in models.items():
  p=LEGACY/f'models/{name}_final.pt';assert sha(p)==record['sha256'][name]
  saved=torch.load(p,map_location='cpu',weights_only=False);assert saved['update']==3216
  model.load_state_dict(saved['state_dict'],strict=True);model.to('cuda:0').eval()
 return models
def validator(seed,n=256,L=65536):
 raw=yaml.safe_load((ROOT/'configs/deepseek_sboed_location.yaml').read_text())['benchmark']
 return CommonValidation(raw,30,seed,n,L,batch=16,chunk=16384,device='cuda:0')
@torch.no_grad()
def contexts(model,val):
 raw=val.raw;bench=make_benchmark(raw);features={s:[] for s in STAGES};top2={s:[] for s in STAGES}
 with torch.device('cuda:0'):
  for start in range(0,val.systems,16):
   n=min(16,val.systems-start);r=rollout(bench,model,30,n,val.seed+start*97451,0,truth=val.truth[start:start+n].cuda(),noise=val.noise[start:start+n].cuda())
   for s in STAGES:
    f=r['features'][s];model(f,s)
    slots=f[:,1:].reshape(n,30,4)
    features[s].extend((slots[:,:s,2]*bench.obs_scale).mean(1).cpu().tolist())
    top2[s].extend(model.last_dispatch['indices'][:,:2].cpu().tolist())
 return {s:np.asarray(v) for s,v in features.items()},{s:np.asarray(v) for s,v in top2.items()}
def holm(rows):
 ordered=sorted(range(len(rows)),key=lambda i:rows[i]['p_approx_two_sided']);maxp=0
 for rank,i in enumerate(ordered):
  maxp=max(maxp,min(1.,(len(rows)-rank)*rows[i]['p_approx_two_sided']));rows[i]['holm_p']=maxp
 return rows
def audit(out):
 models=load_legacy();old=json.loads((LEGACY/'reports/final_development.json').read_text())
 # Check reported summaries directly against preserved raw scores.
 recovered=[]
 for row in old['results']:
  split=row['split'];v1=np.load(LEGACY/f'evaluations/{split}_v1.npy');v2=np.load(LEGACY/f'evaluations/{split}_v2.npy')
  assert np.isfinite(v1).all() and np.isfinite(v2).all()
  assert abs(v2.mean()-row['EIG']['v2']['mean'])<1e-12
  recovered.append({'split':split,'v1':summary(v1),'v2':summary(v2),'v2_minus_v1':summary(v2-v1)})
 fit=validator(97402734,512,0);fitvalues,_=contexts(models['v2'],fit)
 thresholds={s:float(np.median(fitvalues[s])) for s in STAGES};write(out/'frozen_context_thresholds.json',thresholds)
 records=[];effects=[];diagnostic_started=time.perf_counter()
 for split,seed in [('new_development',97402732),('replication',97402733)]:
  v=validator(seed);ctx,top2=contexts(models['v2'],v);values={};meta={}
  for name,p in models.items():
   values[name],meta[name]=v(p);np.save(out/f'{split}_{name}.npy',values[name])
  for intervention in ('masked','uniform'):
   p=copy.deepcopy(models['v2']);p.ablate_routed=intervention=='masked';p.routing_mode='uniform' if intervention=='uniform' else 'learned'
   values[intervention],meta[intervention]=v(p);np.save(out/f'{split}_{intervention}.npy',values[intervention])
  rows=[]
  for stage in STAGES:
   for expert in range(7):
    p=copy.deepcopy(models['v2']);p.substitution=(stage,expert)
    scores,_=v(p);np.save(out/f'{split}_stage{stage}_expert{expert}.npy',scores)
    eligible=~(top2[stage]==expert).any(1);bins=np.zeros(len(scores),dtype=int) if stage==1 else (ctx[stage]>thresholds[stage]).astype(int)
    for context in (0,) if stage==1 else (0,1):
     mask=eligible&(bins==context)
     if mask.sum()<16:continue
     d=scores[mask]-values['v2'][mask];row=dict(split=split,stage=stage,expert=expert,context=context,**summary(d))
     row['p_approx_two_sided']=math.erfc(abs(row['mean']/max(row['MC_SE'],1e-300))/math.sqrt(2));rows.append(row)
  holm(rows);effects.extend(rows)
  records.append({'split':split,'seed':seed,'EIG':{name:summary(x) for name,x in values.items()},'v2_minus_v1':summary(values['v2']-values['v1']),'full_minus_mask':summary(values['v2']-values['masked']),'learned_minus_uniform':summary(values['v2']-values['uniform']),'metadata':meta})
  write(out/'audit_progress.json',{'records':records,'substitution_effects':effects})
  write(out/'status.json',{'state':'AUDITING_EXISTING_CHECKPOINTS','completed_split':split,'pid':os.getpid()})
  if time.perf_counter()-diagnostic_started>1500:break
 write(out/'audit_results.json',{'recovered_original_results':recovered,'new_paired_evaluations':records,'substitution_effects_holm':effects,'diagnostic_GPU_seconds':time.perf_counter()-diagnostic_started,'checkpoint_sha256':json.loads((LEGACY/'metadata/final_checkpoint_freeze.json').read_text())['sha256'],'source':str(LEGACY),'scientific_interpretation':'Prior pilot has no demonstrated v1 EIG improvement. Mask/uniform differences establish frozen component dependence, not independently useful specialist competence. Substitution tests remain finite-resolution exploratory tests.'})
 return records,effects

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['audit','train'],required=True);p.add_argument('--output',required=True);p.add_argument('--audit-from')
 a=p.parse_args(argv);out=Path(a.output).resolve();assert ROOT/'scratch_runs' in out.parents
 out.mkdir(parents=True,exist_ok=False);write(out/'source_manifest.json',source_manifest())
 write(out/'protocol.json',{'action':a.action,'pid':os.getpid(),'maximum_diagnostic_GPU_seconds':1800,'fit_seed':97402734,'development_seed':97402732,'replication_seed':97402733,'reserved_official_seed1001_unused':True,'N':256,'L':65536,'stages':STAGES,'estimand':'Paired terminal information bound under observable-context prefix law; not exact posterior conditional utility','multiplicity':'Holm across expert/stage/context substitution tests; normal-approximation MC p values; independent replication reported'})
 lock=None
 try:
  lock=lock_gpu(out);start=time.perf_counter()
  if a.action=='audit':
   audit(out);write(out/'status.json',{'state':'AUDIT_COMPLETE','pid':os.getpid(),'GPU_locked_seconds':time.perf_counter()-start})
  else:
   assert a.audit_from,'Existing-checkpoint audit must precede new training'
   torch.set_num_threads(1)
   from src.research.moe_v2_context_training import train
   train(out,Path(a.audit_from))
 except Exception as e:
  write(out/'status.json',{'state':'FAILED_PRESERVED','pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc()});raise
 finally:
  if lock:lock.close()

if __name__=='__main__':main()
