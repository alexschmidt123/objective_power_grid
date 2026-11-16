"""Predeclared common-bank EIG and frozen command/competency interventions."""
import copy,time,gc,subprocess,numpy as np,torch
from native_common import *
from restore_v1 import restore,V1,SG,COL
from v2_policy import V2Actor
from soft_policy import CollaborativeActor
from option_policy import parts as option_parts
from moe_actor import DirectDesignHeads
from src.research.vendor.vsoed.utils import Net
from frozen_evaluator import contrast_scores
from v2_tasks import labels
import parent_evaluation as lib
stats=lib.stats;holm=lib.holm;E=O/'evaluation';CFG=json.loads((O/'CONFIG_LOCKED.json').read_text());TEMPLATE=None;TEACHER=None

def template():
 global TEMPLATE
 if TEMPLATE is None:
  state=restore();TEMPLATE=state['algorithm'].actor_net.eval();del state;gc.collect();torch.cuda.empty_cache()
 return TEMPLATE

def actor_load(method):
 if method in ['candidate','stage','prepared','initial_candidate','initial_stage']:
  actor=V2Actor(copy.deepcopy(template()),method in ['stage','initial_stage']);path=O/('preparation/prepared.pt' if method=='prepared' else 'initial/'+method[8:]+'_actor.pt' if method.startswith('initial_') else f'arms/{method}/actor.pt')
 elif method in ['continuation','initial_continuation','v1','v1_control']:
  actor=copy.deepcopy(template());actor.dynamic=method!='v1_control';path=O/('inputs/'+method+'.pt' if method.startswith('v1') else 'initial/continuation_actor.pt' if method.startswith('initial_') else 'arms/continuation/actor.pt')
 elif method=='sg':return copy.deepcopy(template().base).eval().requires_grad_(False),O/'inputs/sg.pt'
 elif method=='sg_control':
  from sg_policy import SGActor
  cp=__import__('json').loads((O/'inputs/COL_CONFIG.json').read_text());base=CollaborativeActor(cp).cuda();base.load_state_dict(torch.load(O/'inputs/parent.pt',map_location='cuda',weights_only=False)['actor_state']);actor=SGActor(base,copy.deepcopy(base),False);path=O/'inputs/sg_control.pt'
 elif method in ['parent','historical_control']:actor=CollaborativeActor(json.loads((O/'inputs/COL_CONFIG.json').read_text())).cuda();path=O/f'inputs/{method}.pt'
 else:
  actor=Net([117,256,256,256,2],torch.nn.ReLU,torch.tensor([[-4.,4.]]*2,device='cuda'),'actor')
  if method!='vsoed':actor.net=torch.nn.Sequential(*list(actor.net.children())[:4],DirectDesignHeads())
  if method=='reference':actor.net[4].fallback_gate=torch.nn.Linear(256,1)
  actor=actor.cuda();path=O/f'inputs/{method}.pt'
 actor.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['actor_state']);return actor.eval().requires_grad_(False),path

def simulate(actor,truth,noise,u,method,collect=False,outside=None,commands=None,gates=None,ordinary_beta=None,disable=None,response_off=None,replace_expert=None,donor_slots=None):
 n=len(truth);acts=[];ys=[];record={k:[] for k in ['X','features','commands','gates','slots','legacy_features','responses','uncertainty_trace']};active=None
 for t in range(30):
  x=lib.native_x(acts,ys,n)
  if method in ['candidate','stage','continuation']:
   c=commands[:,t] if commands is not None and commands.ndim==3 else (commands[t] if commands is not None and commands.ndim==2 else commands);g=gates[:,t] if gates is not None and gates.ndim==3 else (gates[t] if gates is not None else None)
   if ordinary_beta is not None:
    s=actor.history(x);features=torch.cat([torch.ones(n,1,device=x.device),s],-1);g=(features.double()@ordinary_beta).softmax(-1).float()
   extra=dict(response_off=response_off,replace_expert=replace_expert,donor_slots=None if donor_slots is None else donor_slots[:,t]) if isinstance(actor,V2Actor) else {}
   if collect:
    action,info=actor(x,command_override=c,routing_override=g,disable=disable,collect=True,**extra)
    for k in record:record[k].append(x if k=='X' else info[k])
   else:action=actor(x,command_override=c,routing_override=g,disable=disable,**extra)
  elif method=='reference':
   g,proposals=option_parts(actor,x)
   if t%3==0:active=(u[:,t,None]>g.cumsum(-1)).sum(-1).clamp(max=4)
   action=torch.where((active<4)[:,None],proposals.gather(1,active.clamp(max=3)[:,None,None].expand(-1,1,2)).squeeze(1),outside(x))
  else:action=actor(x)
  assert bool(((action>=-4)&(action<=4)).all());distance=(truth.reshape(n,2,2)-action[:,None]).square().sum(-1);y=(.1+(distance+1e-4).reciprocal().sum(-1)).log()+.5*noise[:,t];acts.append(action);ys.append(y[:,None])
 out=dict(actions=acts,observations=ys,truth=truth)
 if collect:out.update({k:torch.stack(v,1) for k,v in record.items()})
 return out

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
 state=restore();a=state['algorithm'];reference=a.actor_net.eval().requires_grad_(False);x=torch.tensor(hist['X'][:,[3,12,24]],device='cuda').reshape(-1,117);cmd=torch.tensor(hist['commands'][:,[3,12,24]],device='cuda').reshape(-1,20);gen=torch.Generator(device='cuda').manual_seed(202611099212);targets=[[] for _ in range(4)];labels_calls=0
 for i in range(0,len(x),128):
  if time.monotonic()>deadline:raise TimeoutError('Fixed competency budget')
  xx=x[i:i+128];_,info=actor(xx,command_override=cmd[i:i+128],collect=True);tt,receipt=labels(a,reference,actor,xx,info,gen,4);labels_calls+=receipt['predictive_observations']
  for k,v in enumerate(tt):targets[k].append(v.cpu().numpy())
 targets=[np.concatenate(v).reshape(512,3,-1) for v in targets];losses={};readout={}
 for method in ['candidate','stage','continuation','prepared','v1']:
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
  rival=min([j for j in range(4) if j!=k],key=lambda j:losses['candidate'][j,k,:256].mean());tests[f'expert{k+1}']=stats((losses['candidate'][rival,k,256:]-losses['candidate'][k,k,256:]).mean(1));retention[f'expert{k+1}']=stats((losses['prepared'][k,k,256:]-losses['candidate'][k,k,256:]).mean(1))
 for j,k in enumerate([1,3]):
  target=targets[k];scale=target[:256].reshape(768,-1).std(0).clip(.1);mu=target[:256].mean((0,1));l0=np.square((target-mu)/scale).mean(-1);lp=np.square((readout['candidate'][j]-target)/scale).mean(-1);constant[f'expert{k+1}']=stats((l0[256:]-lp[256:]).mean(1))
 holm(tests,'Holm4_p');holm(retention,'Holm4_p');holm(constant,'Holm2_p');np.savez_compressed(E/'competency_raw.npz',**{f'target{k}':v for k,v in enumerate(targets)},**{f'{m}_probe_losses':v for m,v in losses.items()});result=dict(designated_advantages=tests,live_vs_prepared=retention,new_readouts_vs_constant=constant,predictive_teacher_observations=labels_calls,analytic_prefix_excluded_from_learned_probes=True,ridge_calibration_only=True,dimension=116,legacy_reference_zero_padding_caveat=True,targets='E2heldoutrotatedhypothesisKL;E4secondstepinformationandtailvalue underapproximatebelief;notexactconditionalEIG');put(E/'COMPETENCY_RESULTS.json',result);del state,a,reference;gc.collect();return result

def run_evaluation(start):
 assert json.loads((O/'evidence/PAIR_COMPLETION.json').read_text())['status']=='COMPLETE';E.mkdir(exist_ok=True);assert not (E/'STARTED.json').exists();tick=time.monotonic();deadline=min(start+32400,tick+3600);put(E/'STARTED.json',dict(time=time.time(),training_updates=0));rng=np.random.default_rng(CFG['evaluation_seed']);bank=dict(truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)));np.savez_compressed(E/'development_validation_bank.npz',**bank);u=np.random.default_rng(CFG['routing_seed']).random((2000,30));np.save(E/'routing_uniforms.npy',u);contrasts=torch.tensor(bank['contrasts'],device='cuda');scores={};methods={};endpoints={m:sha(O/f'arms/{m}/actor.pt') for m in ['candidate','stage','continuation']};status='COMPLETE'
 try:
  with torch.no_grad(),torch.device('cuda'):
   outside,_=actor_load('original')
   for method in ['v1','sg','parent','reference','vsoed','ordinary','historical_control','v1_control','sg_control','continuation','stage','candidate','initial_candidate','initial_stage','initial_continuation']:
    model,_=actor_load(method);scores[method],_,methods[method]=evaluated(model,bank,contrasts,u,method,method,deadline,outside=outside);methods[method]['parameters']=sum(p.numel() for p in model.parameters());methods[method]['unique_buffer_elements']=sum(v.numel() for v in model.buffers());del model
   for method in ['dad','rl_sboed','scorebed']:
    with (E/f'{method}_worker.log').open('wb') as f:r=subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(O/'source/historical_worker.py'),method],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
    if r.returncode:raise RuntimeError(f'{method} evaluation failed')
    scores[method]=np.load(E/f'{method}_scores.npy');methods[method]=json.loads((E/f'{method}_receipt.json').read_text())
   primary={m:stats(scores['candidate']-scores[m]) for m in ['stage','continuation']};holm(primary,'Holm2_p');secondary={m:stats(scores['candidate']-scores[m]) for m in ['v1','sg','reference','vsoed']};holm(secondary,'Holm4_p');adjusted={m:stats((scores['candidate']-scores['initial_candidate'])-(scores[m]-scores[f'initial_{m}'])) for m in ['stage','continuation']};holm(adjusted,'Holm2_p');put(E/'EIG_RESULTS.json',dict(methods=methods,primary=primary,secondary=secondary,initial_adjusted_training_gain=adjusted,bank_sha256=sha(E/'development_validation_bank.npz'),negative_contrasts=1000000,denominator_terms=1000001,finite_contrast_lower_bound=True,frozen_references_not_compute_matched=True))
   rng=np.random.default_rng(CFG['diagnostic_seed']);db=dict(truth=rng.standard_normal((512,4)),noise=rng.standard_normal((512,30)));du=rng.random((512,30));np.savez_compressed(E/'diagnostic_bank.npz',**db,routing_uniforms=du);model,_=actor_load('candidate');base,hist,_=evaluated(model,db,contrasts,du,'candidate','diagnostic_candidate',deadline,collect=True);val={k:v[256:] for k,v in db.items()};vu=du[256:];commands=hist['commands'];gates=hist['gates'];stagec=torch.tensor(commands[:256].mean(0),device='cuda');stageg=torch.tensor(gates[:256].mean(0),device='cuda');rng=np.random.default_rng(CFG['donor_seed']);donor=rng.integers(256,size=(256,30));shuffled=torch.tensor(commands[:256][donor,np.arange(30)[None]],device='cuda');slots=torch.tensor(hist['slots'][:256][donor,np.arange(30)[None]],device='cuda');cmdtests={}
   for name,c in [('fixed_zero',torch.zeros(20,device='cuda')),('stage',stagec),('shuffle',shuffled)]:
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'command_{name}',deadline,commands=c);cmdtests[name]=stats(base[256:]-v)
   holm(cmdtests,'Holm3_p');xx=torch.tensor(hist['X'][:256],device='cuda');ss=model.history(xx).reshape(-1,128).double();C=torch.cat([torch.ones(len(ss),1,device='cuda'),ss],-1);logg=torch.tensor(np.maximum(gates[:256],1e-30),device='cuda').reshape(-1,4).double().log();beta=torch.linalg.solve(C.T@C+torch.eye(129,device='cuda'),C.T@logg);coord={}
   for name,opts in [('uniform',dict(gates=torch.full((30,4),.25,device='cuda'))),('stage',dict(gates=stageg)),('linear',dict(ordinary_beta=beta))]:
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'coord_{name}',deadline,**opts);coord[name]=stats(base[256:]-v)
   holm(coord,'Holm3_p');removal={};donor_tests={};mechanism={}
   for k in range(4):
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'expert{k+1}_removed',deadline,disable=k);removal[str(k+1)]=stats(base[256:]-v)
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'expert{k+1}_donor',deadline,replace_expert=k,donor_slots=slots);donor_tests[str(k+1)]=stats(base[256:]-v)
   holm(removal,'Holm4_p');holm(donor_tests,'Holm4_p')
   for name,k in [('E2off',1),('E4off',3),('alloff',[0,1,2,3])]:
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'response_{name}',deadline,response_off=k);mechanism[name]=stats(base[256:]-v)
   holm(mechanism,'Holm3_p');tasks=competency(model,hist,deadline);put(E/'SPECIALIZATION_RESULTS.json',dict(competency=tasks,expert_removal=removal,expert_donor=donor_tests,response_mechanism=mechanism,intervention_distribution_shift_caveat=True));put(E/'COMMANDER_RESULTS.json',dict(commands=cmdtests,coordination=coord,trained_stage_control_primary=primary['stage'],within_stage_command_SD=commands.std(0).mean(0).tolist(),within_stage_gate_SD=gates.std(0).mean(0).tolist(),linear_projection_is_not_EIG_trained=True));from conditional_diagnostics import run_conditional;run_conditional(model,hist,deadline);timings={}
   for method in ['candidate','stage','continuation','v1','sg']:
    aa,_=actor_load(method);xx=torch.tensor(hist['X'][:1],device='cuda');torch.cuda.synchronize();beg=time.perf_counter()
    for repeat in range(10):
     for t in range(30):aa(xx[:,t])
    torch.cuda.synchronize();timings[method]=dict(milliseconds_per_design=(time.perf_counter()-beg)*1000/300,includes_all_frozen_provider_and_predictive_query_work=True);del aa
   put(E/'LATENCY.json',timings)
 except Exception as exc:
  import traceback
  traceback.print_exc();status='INCOMPLETE';put(E/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
 finally:
  assert all(sha(O/f'arms/{m}/actor.pt')==h for m,h in endpoints.items());put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-tick,main_policies_completed=list(scores),training_updates=0,endpoints_unchanged=True));print('EVALUATION_END',status,flush=True)
