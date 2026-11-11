import copy,gc,subprocess
from common import *
from v3_policy import V3Actor
from frozen_evaluator import contrast_scores
import evaluate_v2 as old_eval
from prepare import conditions

def load_model(method):
 if method in ['candidate','stage','initial']:
  parent,_=old_eval.actor_load('candidate');model=V3Actor(parent,method=='stage').cuda();path=O/('initial/candidate.pt' if method=='initial' else f'arms/{method}/actor.pt');model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['actor_state']);return model.eval().requires_grad_(False)
 old={'hec_v2':'candidate','hec_v1':'v1','sg':'sg','option':'reference','collaborative':'parent','ordinary':'ordinary','vsoed':'vsoed'}[method];return old_eval.actor_load(old)[0]

@torch.no_grad()
def rollout(model,truth,noise,u,eps,method,collect=False,fixed=None,gates=None,remove=None,model_features=None):
 n=len(truth);acts=[];ys=[];hist={k:[] for k in ['X','gates','proposals','model_features','selected','posterior_responsibilities','ds','ys']}
 if method not in ['candidate','stage','initial']:
  outside=old_eval.actor_load('original')[0] if method=='option' else None
  return old_eval.simulate(model,truth,noise,u,{'hec_v2':'candidate','hec_v1':'v1','sg':'sg','option':'reference','collaborative':'parent','ordinary':'ordinary','vsoed':'vsoed'}[method],outside=outside)
 for t in range(30):
  x=old_eval.lib.native_x(acts,ys,n);gg=gates[:,t] if gates is not None and gates.ndim==3 else gates[t] if gates is not None and gates.ndim==2 else gates;mf=model_features[:,t] if model_features is not None else None
  if remove is not None:
   g,_,_=model.parts(x);g[:,remove]=0;gg=g/g.sum(-1,keepdim=True)
  action,info=model.draw(x,u[:,t],eps[:,t],fixed=fixed,gates=gg,model_features_override=mf);assert bool((action.abs()<4).all());y=mean_signal(truth[:,None],action[:,None])[:,0,0]+.5*noise[:,t];acts.append(action);ys.append(y[:,None])
  if collect:
   _,resp=model.log_prob(action,info['gates'],info['means'])
   for k in hist:hist[k].append(x if k=='X' else action if k=='ds' else y[:,None] if k=='ys' else resp if k=='posterior_responsibilities' else info[k])
 result=dict(actions=acts,observations=ys,truth=truth)
 if collect:result.update({k:torch.stack(v,1) for k,v in hist.items()})
 return result

def evaluated(model,bank,contrasts,method,label,deadline,collect=False,**options):
 E=O/'evaluation';score=[];curves=[];hist={};start=time.monotonic();bench=type('Benchmark',(),{'sigma':.5,'calls':0})()
 with torch.no_grad():
  for i in range(0,len(bank['truth']),16):
   if time.monotonic()>deadline:raise TimeoutError('Predeclared evaluation deadline')
   sl=slice(i,i+16);opts={k:(v[sl] if torch.is_tensor(v) and v.ndim>=3 else v) for k,v in options.items()};data=rollout(model,*[torch.tensor(bank[k][sl],device='cuda') for k in ['truth','noise','u','eps']],method,collect=collect,**opts);v,_=contrast_scores(bench,data,contrasts,16384,'cuda');score.extend(v[:,-1].cpu().numpy());curves.append(v.cpu().numpy())
   if collect:
    for k in ['X','gates','proposals','model_features','selected','posterior_responsibilities','ds','ys']:hist.setdefault(k,[]).append(data[k].cpu().numpy())
 values=np.asarray(score);np.save(E/f'{label}_scores.npy',values);np.save(E/f'{label}_curves.npy',np.concatenate(curves));hist={k:np.concatenate(v) for k,v in hist.items()}
 if hist:np.savez_compressed(E/f'{label}_histories.npz',**hist)
 receipt=dict(**stats(values),seconds=time.monotonic()-start,simulator_observations=len(values)*30,negative_density_calls=bench.calls,positive_density_calls=len(values)*30,parameters=sum(p.numel() for p in model.parameters()),frozen_buffer_elements=sum(v.numel() for v in model.buffers()));put(E/f'{label}_receipt.json',receipt);print('EVALUATED',label,json.dumps(receipt),flush=True);return values,hist,receipt

def run(start):
 E=O/'evaluation';E.mkdir();tick=time.monotonic();deadline=min(start+CFG['training_grace_seconds'],tick+CFG['evaluation_max_seconds']);rng=np.random.default_rng(CFG['evaluation_seed']);bank=dict(truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)),u=rng.random((2000,30)),eps=rng.standard_normal((2000,30,2)));np.savez_compressed(E/'development_validation_bank.npz',**bank);neg=torch.tensor(bank['contrasts'],device='cuda');scores={};methods={};endpoints={m:sha(O/f'arms/{m}/actor.pt') for m in ['candidate','stage']};status='INCOMPLETE'
 try:
  for m in ['hec_v2','hec_v1','sg','option','collaborative','ordinary','vsoed','initial','stage','candidate']:
   model=load_model(m);scores[m],_,methods[m]=evaluated(model,bank,neg,m,m,deadline);del model;gc.collect();torch.cuda.empty_cache()
  for m in ['dad','rl_sboed','scorebed']:
   with (E/f'{m}_worker.log').open('wb') as f:p=subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(O/'source/historical_worker.py'),m],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'MOE_V2_RUN_ROOT':str(O),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'},timeout=max(1,deadline-time.monotonic()))
   assert p.returncode==0,f'{m} frozen evaluation failed';scores[m]=np.load(E/f'{m}_scores.npy');methods[m]=json.loads((E/f'{m}_receipt.json').read_text())
  primary=holm({m:stats(scores['candidate']-scores[m]) for m in ['stage','hec_v2']});secondary=holm({m:stats(scores['candidate']-scores[m]) for m in ['hec_v1','sg','option','vsoed']});put(E/'EIG_RESULTS.json',dict(methods=methods,primary=primary,secondary=secondary,initial_change_vs_frozen=stats(scores['initial']-scores['hec_v2']),new_bank_sha256=sha(E/'development_validation_bank.npz'),finite_contrast_lower_bound=True,training_lineages=1,frozen_baselines_not_compute_matched=True));diagnostics(neg,deadline);status='COMPLETE'
 except Exception as exc:
  import traceback
  traceback.print_exc();put(E/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
 finally:
  assert all(sha(O/f'arms/{m}/actor.pt')==h for m,h in endpoints.items());put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-tick,completed_policies=list(scores),no_training_updates=True,checkpoints_unchanged=True));print('EVALUATION_END',status,flush=True)

def diagnostics(neg,deadline):
 E=O/'evaluation';rng=np.random.default_rng(CFG['diagnostic_seed']);bank=dict(truth=rng.standard_normal((512,4)),noise=rng.standard_normal((512,30)),u=rng.random((512,30)),eps=rng.standard_normal((512,30,2)));np.savez_compressed(E/'diagnostic_bank.npz',**bank);model=load_model('candidate');base,hist,_=evaluated(model,bank,neg,'candidate','diagnostic',deadline,collect=True);val={k:v[256:] for k,v in bank.items()};gates=hist['gates'];tests={};stageg=torch.tensor(gates[:256].mean(0),device='cuda');donor=rng.integers(256,size=(256,30));shuffled=torch.tensor(gates[:256][donor,np.arange(30)[None]],device='cuda');features=torch.tensor(hist['model_features'][:256][donor,np.arange(30)[None]],device='cuda')
 for name,opts in [('uniform',dict(gates=torch.full((30,4),.25,device='cuda'))),('stage_mean',dict(gates=stageg)),('shuffled_gates',dict(gates=shuffled)),('shuffled_proposal_features',dict(model_features=features))]:
  vv,_,_=evaluated(model,val,neg,'candidate',name,deadline,**opts);tests[name]=stats(base[256:]-vv)
 holm(tests);fixed={};removed={}
 for k in range(4):
  vv,_,_=evaluated(model,val,neg,'candidate',f'fixed_expert{k}',deadline,fixed=k);fixed[str(k)]=stats(base[256:]-vv);vv,_,_=evaluated(model,val,neg,'candidate',f'remove_expert{k}',deadline,remove=k);removed[str(k)]=stats(base[256:]-vv)
 holm(fixed);holm(removed);props=hist['proposals'][256:];distance=np.zeros((4,4));corr=np.eye(4)
 for i in range(4):
  for j in range(4):distance[i,j]=np.linalg.norm(props[:,:,i]-props[:,:,j],axis=-1).mean();corr[i,j]=np.corrcoef(props[:,:,i].flatten(),props[:,:,j].flatten())[0,1]
 prob=hist['posterior_responsibilities'][256:];routing=dict(interventions=tests,fixed_expert_comparisons=fixed,expert_removal=removed,within_stage_gate_SD=gates[256:].std(0).mean(0).tolist(),selection_counts=np.bincount(hist['selected'][256:].flatten(),minlength=4).tolist(),selection_frequency=(np.bincount(hist['selected'][256:].flatten(),minlength=4)/hist['selected'][256:].size).tolist(),proposal_distance=distance.tolist(),proposal_output_correlations=corr.tolist(),mean_action_responsibility_entropy=float(-(prob*np.log(np.maximum(prob,1e-30))).sum(-1).mean()),action_differences_are_not_competence=True,intervention_distribution_shift_caveat=True);put(E/'ROUTING_AND_EXPERT_RESULTS.json',routing);from conditional import run_conditional;run_conditional(model,bank,hist,deadline);latency={}
 for m in ['candidate','stage','hec_v2']:
  mm=load_model(m);x=torch.tensor(hist['X'][:1,12],device='cuda');u=torch.tensor([.43],device='cuda');e=torch.zeros(1,2,device='cuda')
  with torch.no_grad():
   for _ in range(5):mm.draw(x,u,e) if m!='hec_v2' else mm(x)
   torch.cuda.synchronize();t=time.perf_counter()
   for _ in range(100):mm.draw(x,u,e) if m!='hec_v2' else mm(x)
   torch.cuda.synchronize();latency[m]=(time.perf_counter()-t)*10
 put(E/'LATENCY.json',dict(milliseconds_per_design=latency,includes_frozen_provider=True))
