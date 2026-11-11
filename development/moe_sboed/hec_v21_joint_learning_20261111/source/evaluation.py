import copy,gc,subprocess,time,numpy as np,torch
from common import *
from joint_policy import JointHEC
from v2_policy import V2Actor
from v2_tasks import labels
import evaluate_v2 as legacy
from frozen_evaluator import contrast_scores
E=O/'evaluation';lib=legacy.lib;stats=legacy.stats;holm=legacy.holm;TEMPLATE=None

def parent():
 global TEMPLATE
 if TEMPLATE is None:TEMPLATE=legacy.actor_load('candidate')[0]
 return copy.deepcopy(TEMPLATE)
def actor_load(method):
 if method in ['candidate','continuation']:
  model=JointHEC(parent()) if method=='candidate' else parent();path=O/f'arms/{method}/actor.pt';model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['actor_state']);return model.eval().requires_grad_(False),path
 if method=='original':return parent(),OLD/'arms/candidate/actor.pt'
 return legacy.actor_load('ordinary' if method=='conventional' else method)
def simulate(actor,truth,noise,u,method,**kwargs):return legacy.simulate(actor,truth,noise,u,'ordinary' if method=='conventional' else 'vsoed' if method=='vsoed' else 'candidate',**kwargs)

def evaluated(actor,bank,contrasts,u,method,label,deadline,collect=False,**kwargs):
 vals=[];curves=[];hist={};tick=time.monotonic();bench=type('Benchmark',(),{'sigma':.5,'calls':0})()
 for i in range(0,len(bank['truth']),16):
  if time.monotonic()>deadline:raise TimeoutError('Fixed evaluation budget')
  sl=slice(i,i+16);opts=dict(kwargs)
  for key in ['commands','gates','donor_slots']:
   if opts.get(key) is not None and opts[key].ndim in [3,4]:opts[key]=opts[key][sl]
  d=simulate(actor,torch.tensor(bank['truth'][sl],device='cuda'),torch.tensor(bank['noise'][sl],device='cuda'),torch.tensor(u[sl],device='cuda'),method,collect=collect,**opts);v,_=contrast_scores(bench,d,contrasts,16384,'cuda');vals.extend(v[:,-1].cpu().numpy());curves.append(v.cpu().numpy())
  if collect:
   data={k:d[k].cpu().numpy() for k in ['X','features','commands','gates','slots','legacy_features','responses','uncertainty_trace']};data.update(ds=torch.stack(d['actions'],1).cpu().numpy(),ys=torch.stack(d['observations'],1).cpu().numpy(),truth=d['truth'].cpu().numpy())
   for k,v in data.items():hist.setdefault(k,[]).append(v)
 vals=np.array(vals);np.save(E/f'{label}_scores.npy',vals);np.save(E/f'{label}_curves.npy',np.concatenate(curves));hist={k:np.concatenate(v) for k,v in hist.items()}
 if hist:np.savez_compressed(E/f'{label}_histories.npz',**hist)
 receipt=dict(**stats(vals),seconds=time.monotonic()-tick,simulator_observations=len(vals)*30,negative_density_calls=bench.calls,positive_density_calls=len(vals)*30);put(E/f'{label}_receipt.json',receipt);print('EVALUATED',label,json.dumps(receipt),flush=True);return vals,hist,receipt

def competency(actor,hist,deadline):
 state=restore();a=state['algorithm'];reference=a.actor_net.eval().requires_grad_(False);x=torch.tensor(hist['X'][:,[3,12,24]],device='cuda').reshape(-1,117);cmd=torch.tensor(hist['commands'][:,[3,12,24]],device='cuda').reshape(-1,20);gen=torch.Generator(device='cuda').manual_seed(CFG['diagnostic_seed']+1);targets=[[] for _ in range(4)];labels_calls=0
 for i in range(0,len(x),128):
  if time.monotonic()>deadline:raise TimeoutError('Fixed competency budget')
  xx=x[i:i+128];_,info=actor(xx,command_override=cmd[i:i+128],collect=True);tt,receipt=labels(a,reference,actor,xx,info,gen,4);labels_calls+=receipt['predictive_observations']
  for k,v in enumerate(tt):targets[k].append(v.cpu().numpy())
 targets=[np.concatenate(v).reshape(512,3,-1) for v in targets];losses={};readout={}
 for method in ['candidate','continuation','original']:
  model,_=actor_load(method);features=[];pred2=[];pred4=[]
  for i in range(0,len(x),128):
   xx=x[i:i+128];_,di=model(xx,command_override=cmd[i:i+128],collect=True)
   if isinstance(model,V2Actor):
    # Exclude the 12 exact analytic bypass coordinates when testing learned representations.
    z=torch.cat([di['slots'][...,:64],di['responses'][...,12:]],-1);pred2.append(model.response_readouts[0](di['responses'][:,1,12:]).cpu().numpy());pred4.append(model.response_readouts[1](di['responses'][:,3,12:]).cpu().numpy())
   else:z=torch.cat([di['features'],torch.zeros(len(xx),4,52,device='cuda')],-1)
   features.append(z.cpu().numpy())
  Z=np.concatenate(features).reshape(512,3,4,116);L=np.zeros((4,4,512,3))
  for expert in range(4):
   cal=Z[:256,:,expert].reshape(-1,116).astype(float);mu=cal.mean(0);sd=cal.std(0).clip(1e-6);C=np.column_stack([np.ones(768),(cal-mu)/sd]);F=np.column_stack([np.ones(1536),(Z[:,:,expert].reshape(-1,116)-mu)/sd]);inverse=np.linalg.solve(C.T@C+np.eye(117),C.T)
   for k,target in enumerate(targets):
    scale=target[:256].reshape(768,-1).std(0).clip(.1);prediction=(F@(inverse@target[:256].reshape(768,-1))).reshape(512,3,-1);L[expert,k]=np.square((prediction-target)/scale).mean(-1)
  losses[method]=L
  if pred2:readout[method]=[np.concatenate(pred2).reshape(512,3,-1),np.concatenate(pred4).reshape(512,3,-1)]
  del model
 tests={};retention={};constant={}
 for k in range(4):
  rival=min([j for j in range(4) if j!=k],key=lambda j:losses['candidate'][j,k,:256].mean());tests[f'expert{k+1}']=stats((losses['candidate'][rival,k,256:]-losses['candidate'][k,k,256:]).mean(1));retention[f'expert{k+1}']=stats((losses['original'][k,k,256:]-losses['candidate'][k,k,256:]).mean(1))
 for j,k in enumerate([1,3]):
  target=targets[k];scale=target[:256].reshape(768,-1).std(0).clip(.1);mu=target[:256].mean((0,1));l0=np.square((target-mu)/scale).mean(-1);lp=np.square((readout['candidate'][j]-target)/scale).mean(-1);constant[f'expert{k+1}']=stats((l0[256:]-lp[256:]).mean(1))
 holm(tests,'Holm4_p');holm(retention,'Holm4_p');holm(constant,'Holm2_p');np.savez_compressed(E/'competency_raw.npz',**{f'target{k}':v for k,v in enumerate(targets)},**{f'{m}_probe_losses':v for m,v in losses.items()});result=dict(designated_advantages=tests,live_vs_original=retention,new_readouts_vs_constant=constant,predictive_teacher_observations=labels_calls,analytic_prefix_excluded_from_learned_probes=True,ridge_calibration_only=True,dimension=116,legacy_reference_zero_padding_caveat=True,targets='E2heldoutrotatedhypothesisKL;E4secondstepinformationandtailvalue underapproximatebelief;notexactconditionalEIG');put(E/'COMPETENCY_RESULTS.json',result);del state,a,reference;gc.collect();return result

def run_evaluation(start):
 assert json.loads((O/'evidence/PAIR_COMPLETION.json').read_text())['status']=='COMPLETE';E.mkdir();tick=time.monotonic();deadline=min(start+CFG['training_grace_seconds'],tick+CFG['evaluation_max_seconds']);rng=np.random.default_rng(CFG['evaluation_seed']);bank=dict(truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)));np.savez_compressed(E/'development_validation_bank.npz',**bank);u=rng.random((2000,30));np.save(E/'routing_uniforms.npy',u);neg=torch.tensor(bank['contrasts'],device='cuda');scores={};methods={};endpoint={k:sha(O/f'arms/{k}/actor.pt') for k in ['candidate','continuation']};status='INCOMPLETE'
 try:
  for name in ['original','conventional','vsoed','continuation','candidate','uniform','expert1','expert2','expert3','expert4']:
   model,_=actor_load('candidate' if name in ['uniform','expert1','expert2','expert3','expert4'] else name);opts={}
   if name=='uniform':opts['gates']=torch.full((30,4),.25,device='cuda')
   elif name.startswith('expert'):opts['gates']=torch.eye(4,device='cuda')[int(name[-1])-1][None].expand(30,-1)
   scores[name],_,methods[name]=evaluated(model,bank,neg,u,'candidate' if name in ['uniform','expert1','expert2','expert3','expert4'] else name,name,deadline,**opts);methods[name]['parameters']=sum(p.numel() for p in model.parameters());methods[name]['frozen_buffer_elements']=sum(v.numel() for v in model.buffers());del model;gc.collect();torch.cuda.empty_cache()
  for name in ['dad','rl_sboed','scorebed']:
   with (E/f'{name}_worker.log').open('wb') as f:p=subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(O/'source/historical_worker.py'),name],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'MOE_V2_RUN_ROOT':str(O),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'},timeout=max(1,deadline-time.monotonic()))
   assert p.returncode==0,f'{name} frozen evaluation failed';scores[name]=np.load(E/f'{name}_scores.npy');methods[name]=json.loads((E/f'{name}_receipt.json').read_text())
  primary={name:stats(scores['candidate']-scores[name]) for name in ['continuation','uniform','expert1','expert2','expert3','expert4']};holm(primary,'Holm6_p');best=max(['expert1','expert2','expert3','expert4'],key=lambda k:methods[k]['mean']);secondary={k:stats(scores['candidate']-scores[k]) for k in ['original','conventional']};holm(secondary,'Holm2_p');external={k:stats(scores['candidate']-scores[k]) for k in ['vsoed','dad','rl_sboed','scorebed']};holm(external,'Holm4_p');put(E/'EIG_RESULTS.json',dict(methods=methods,primary=primary,best_observed_expert=best,full_minus_best=primary[best],secondary=secondary,external_baselines=external,bank_sha256=sha(E/'development_validation_bank.npz'),finite_contrast_lower_bound=True,single_training_lineage=True,external_baselines_not_compute_matched=True));diagnostics(neg,deadline);status='COMPLETE'
 except Exception as exc:
  import traceback
  traceback.print_exc();put(E/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
 finally:
  assert all(sha(O/f'arms/{k}/actor.pt')==v for k,v in endpoint.items());put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-tick,policies_completed=list(scores),no_training_updates=True,checkpoints_unchanged=True));print('EVALUATION_END',status,flush=True)

def diagnostics(neg,deadline):
 rng=np.random.default_rng(CFG['diagnostic_seed']);db=dict(truth=rng.standard_normal((512,4)),noise=rng.standard_normal((512,30)));du=rng.random((512,30));np.savez_compressed(E/'diagnostic_bank.npz',**db,routing_uniforms=du);actor,_=actor_load('candidate');base,hist,_=evaluated(actor,db,neg,du,'candidate','diagnostic_candidate',deadline,collect=True);val={k:v[256:] for k,v in db.items()};tests={}
 for k in range(4):
  score,_,_=evaluated(actor,val,neg,du[256:],'candidate',f'diagnostic_remove_expert{k+1}',deadline,disable=k);tests[str(k+1)]=stats(base[256:]-score)
 holm(tests,'Holm4_p');hook=actor.relevance.register_forward_hook(lambda module,args,output:torch.zeros_like(output))
 try:off,_,_=evaluated(actor,val,neg,du[256:],'candidate','diagnostic_new_correction_off',deadline)
 finally:hook.remove()
 skills=competency(actor,hist,deadline);g=hist['gates'][256:];put(E/'EXPERT_AND_ROUTING_RESULTS.json',dict(expert_removal=tests,new_correction_frozen_effect=stats(base[256:]-off),within_stage_gate_SD=g.std(0).mean(0).tolist(),gate_means=g.mean((0,1)).tolist(),competency=skills,frozen_ablation_not_training_causality=True));from conditional_diagnostics import run_conditional;run_conditional(actor,hist,deadline);latency={}
 for name in ['candidate','continuation','original','conventional']:
  model,_=actor_load(name);x=torch.tensor(hist['X'][:1,12],device='cuda')
  with torch.no_grad():
   for _ in range(5):model(x)
   torch.cuda.synchronize();t=time.perf_counter()
   for _ in range(100):model(x)
   torch.cuda.synchronize();latency[name]=(time.perf_counter()-t)*10
 put(E/'LATENCY.json',dict(milliseconds_per_design=latency,history_stage=12,repeats=100,includes_all_frozen_provider_queries=True))
