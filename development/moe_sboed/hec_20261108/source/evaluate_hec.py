"""Predeclared common-bank EIG and frozen command/competency interventions."""
import copy,time,gc,subprocess,numpy as np,torch
from native_common import *
from restore_sg import restore,SG,COL
from prepare_specialists import create_model
from soft_policy import CollaborativeActor
from option_policy import parts as option_parts
from moe_actor import DirectDesignHeads
from src.research.vendor.vsoed.utils import Net
from frozen_evaluator import contrast_scores
from specialist_tasks import targets
import parent_evaluation as lib
stats=lib.stats;holm=lib.holm;E=O/'evaluation';CFG=json.loads((O/'CONFIG_LOCKED.json').read_text());TEMPLATE=None

def template():
 global TEMPLATE
 if TEMPLATE is None:
  state=restore();TEMPLATE=create_model(state['algorithm']).eval();del state;gc.collect();torch.cuda.empty_cache()
 return TEMPLATE

def actor_load(method):
 if method in ['candidate','control','prepared']:
  actor=copy.deepcopy(template());actor.dynamic=method!='control';p=O/('preparation/prepared.pt' if method=='prepared' else f'arms/{method}/actor.pt')
 elif method=='sg':return copy.deepcopy(template().base).eval(),O/'inputs/sg.pt'
 elif method in ['parent','historical_control']:actor=CollaborativeActor(json.loads((O/'inputs/COL_CONFIG.json').read_text())).cuda();p=O/f'inputs/{method}.pt'
 else:
  actor=Net([117,256,256,256,2],torch.nn.ReLU,torch.tensor([[-4.,4.]]*2,device='cuda'),'actor')
  if method!='vsoed':actor.net=torch.nn.Sequential(*list(actor.net.children())[:4],DirectDesignHeads())
  if method=='reference':actor.net[4].fallback_gate=torch.nn.Linear(256,1)
  actor=actor.cuda();p=O/f'inputs/{method}.pt'
 actor.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['actor_state']);return actor.eval().requires_grad_(False),p

def simulate(actor,truth,noise,u,method,collect=False,outside=None,commands=None,gates=None,ordinary_beta=None,disable=None):
 n=len(truth);acts=[];ys=[];record={k:[] for k in ['X','features','commands','gates']};active=None
 for t in range(30):
  x=lib.native_x(acts,ys,n)
  if method in ['candidate','control']:
   c=commands[:,t] if commands is not None and commands.ndim==3 else (commands[t] if commands is not None and commands.ndim==2 else commands);g=gates[:,t] if gates is not None and gates.ndim==3 else (gates[t] if gates is not None else None)
   if ordinary_beta is not None:
    s=actor.history(x);features=torch.cat([torch.ones(n,1,device=x.device),s],-1);g=(features.double()@ordinary_beta).softmax(-1).float()
   if collect:
    action,info=actor(x,command_override=c,routing_override=g,disable=disable,collect=True)
    for k in record:record[k].append(x if k=='X' else info[k])
   else:action=actor(x,command_override=c,routing_override=g,disable=disable)
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
  for key in ['commands','gates']:
   if opts.get(key) is not None and opts[key].ndim==3:opts[key]=opts[key][sl]
  d=simulate(actor,torch.tensor(bank['truth'][sl],device='cuda'),torch.tensor(bank['noise'][sl],device='cuda'),torch.tensor(u[sl],device='cuda'),method,collect=collect,**opts);v,_=contrast_scores(bench,d,contrasts,16384,'cuda');vals.extend(v[:,-1].cpu().numpy());curves.append(v.cpu().numpy())
  if collect:
   data={k:d[k].cpu().numpy() for k in ['X','features','commands','gates']};data.update(ds=torch.stack(d['actions'],1).cpu().numpy(),ys=torch.stack(d['observations'],1).cpu().numpy(),truth=d['truth'].cpu().numpy())
   for k,v in data.items():hist.setdefault(k,[]).append(v)
 vals=np.array(vals);np.save(E/f'{label}_scores.npy',vals);np.save(E/f'{label}_curves.npy',np.concatenate(curves));hist={k:np.concatenate(v) for k,v in hist.items()}
 if hist:np.savez_compressed(E/f'{label}_histories.npz',**hist)
 receipt=dict(**stats(vals),seconds=time.monotonic()-tick,simulator_observations=len(vals)*30,negative_density_calls=bench.calls,positive_density_calls=len(vals)*30);put(E/f'{label}_receipt.json',receipt);print('EVALUATED',label,json.dumps(receipt),flush=True);return vals,hist,receipt

def competency(actor,hist,deadline):
 x=torch.tensor(hist['X'][:,[3,12,24]],device='cuda').reshape(-1,117);labels=[[] for _ in range(4)];Z={m:[] for m in ['candidate','control','prepared','sg']};readloss={m:[] for m in ['candidate','control','prepared']};gen=torch.Generator(device='cuda').manual_seed(202611099109)
 for i in range(0,len(x),128):
  if time.monotonic()>deadline:raise TimeoutError('Competency fixed deadline')
  xx=x[i:i+128];_,info=actor(xx,collect=True);tt,receipt=targets(actor,xx,info,gen)
  for k in range(4):labels[k].append(tt[k].cpu().numpy())
 # Shared command schedule isolates representations from changing teacher queries.
 commands=torch.tensor(hist['commands'][:,[3,12,24]],device='cuda').reshape(-1,20)
 for method in Z:
  model,_=actor_load(method);zs=[];ls=[]
  for i in range(0,len(x),128):
   xx=x[i:i+128]
   if method=='sg':zz=model.parts(xx)[1]
   else:_,di=model(xx,command_override=commands[i:i+128],collect=True);zz=di['features'];ls.append([h(zz[:,k]).cpu().numpy() for k,h in enumerate(model.readouts)])
   zs.append(zz.cpu().numpy())
  Z[method]=np.concatenate(zs).reshape(512,3,4,64)
  if method!='sg':readloss[method]=[np.concatenate([r[k] for r in ls]).reshape(512,3,-1) for k in range(4)]
  del model
 labels=[np.concatenate(v).reshape(512,3,-1) for v in labels];allloss={};probe_tests={};rivals=[]
 for method,z in Z.items():
  loss=np.zeros((4,4,512,3))
  for expert in range(4):
   cal=z[:256,:,expert].reshape(-1,64).astype(float);mu=cal.mean(0);sd=cal.std(0).clip(1e-6);C=np.column_stack([np.ones(len(cal)),(cal-mu)/sd]);F=np.column_stack([np.ones(1536),(z[:,:,expert].reshape(-1,64)-mu)/sd]);inverse=np.linalg.solve(C.T@C+np.eye(65),C.T)
   for k in range(4):
    target=labels[k];scale=target[:256].reshape(-1,target.shape[-1]).std(0).clip(.1);beta=inverse@target[:256].reshape(768,-1);prediction=(F@beta).reshape(512,3,-1);loss[expert,k]=np.square((prediction-target)/scale).mean(-1)
  allloss[method]=loss
 for k in range(4):
  competitors=[j for j in range(4) if j!=k];rival=min(competitors,key=lambda j:allloss['candidate'][j,k,:256].mean());rivals.append(rival);probe_tests[f'expert{k+1}_designated_advantage']=stats((allloss['candidate'][rival,k,256:]-allloss['candidate'][k,k,256:]).mean(1))
 holm(probe_tests,'Holm4_p');retention={};constant={};raw={}
 for k,target in enumerate(labels):
  scale=target[:256].reshape(-1,target.shape[-1]).std(0).clip(.1);mu=target[:256].mean((0,1));constantloss=np.square((mu-target)/scale).mean(-1);pred=readloss['candidate'][k];loss=np.square((pred-target)/scale).mean(-1);constant[f'expert{k+1}_beats_constant']=stats((constantloss[256:]-loss[256:]).mean(1));retention[f'expert{k+1}_live_vs_prepared_probe']=stats((allloss['prepared'][k,k,256:]-allloss['candidate'][k,k,256:]).mean(1));raw[f'target{k}']=target
 holm(constant,'Holm4_p');holm(retention,'Holm4_p');np.savez_compressed(E/'competency_raw.npz',**raw,**{f'{m}_probe_losses':v for m,v in allloss.items()});receipt=dict(designated_expert_advantages=probe_tests,frozen_readout_vs_constant=constant,live_vs_prepared_probe=retention,validation='256 heldout histories;three stages clustered by history;ridge1;calibration-selected rival',task4_limit='two-replicate approximate-belief native bootstrap targets;not exact remaining EIG',roles_assigned_by_mathematical_inputs=True);put(E/'COMPETENCY_RESULTS.json',receipt);return receipt

def run_evaluation(start):
 assert json.loads((O/'evidence/PAIR_COMPLETION.json').read_text())['status']=='COMPLETE';E.mkdir(exist_ok=True);assert not (E/'STARTED.json').exists();tick=time.monotonic();deadline=min(start+21600,tick+3600);put(E/'STARTED.json',dict(time=time.time(),training_updates=0));rng=np.random.default_rng(CFG['evaluation_seed']);bank=dict(truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)));np.savez_compressed(E/'development_validation_bank.npz',**bank);u=np.random.default_rng(CFG['routing_seed']).random((2000,30));np.save(E/'routing_uniforms.npy',u);contrasts=torch.tensor(bank['contrasts'],device='cuda');scores={};methods={};endpoints={m:sha(O/f'arms/{m}/actor.pt') for m in ['candidate','control']};status='COMPLETE'
 try:
  with torch.no_grad(),torch.device('cuda'):
   outside,_=actor_load('original')
   for method in ['sg','parent','reference','vsoed','ordinary','historical_control','control','candidate']:
    model,_=actor_load(method);scores[method],_,methods[method]=evaluated(model,bank,contrasts,u,method,method,deadline,outside=outside);methods[method]['network_parameters']=sum(p.numel() for p in model.parameters());methods[method]['frozen_buffers']=sum(v.numel() for v in model.buffers());del model
   for method in ['dad','rl_sboed','scorebed']:
    with (E/f'{method}_worker.log').open('wb') as f:r=subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(O/'source/historical_worker.py'),method],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
    if r.returncode:raise RuntimeError(f'{method} evaluation failed')
    scores[method]=np.load(E/f'{method}_scores.npy');methods[method]=json.loads((E/f'{method}_receipt.json').read_text())
   primary=stats(scores['candidate']-scores['control']);secondary={m:stats(scores['candidate']-scores[m]) for m in ['sg','reference','vsoed','parent']};holm(secondary,'Holm4_p');put(E/'EIG_RESULTS.json',dict(methods=methods,primary_candidate_minus_control=primary,secondary=secondary,other_comparisons_descriptive={m:stats(scores['candidate']-scores[m]) for m in ['dad','rl_sboed','scorebed','ordinary','historical_control']},bank_sha256=sha(E/'development_validation_bank.npz'),negative_prior_contrasts=1000000,total_denominator_terms=1000001,finite_contrast_lower_bound=True,not_compute_matched_against_frozen_references=True))
   rng=np.random.default_rng(CFG['diagnostic_seed']);db=dict(truth=rng.standard_normal((512,4)),noise=rng.standard_normal((512,30)));du=rng.random((512,30));np.savez_compressed(E/'diagnostic_bank.npz',**db,routing_uniforms=du);model,_=actor_load('candidate');base,hist,_=evaluated(model,db,contrasts,du,'candidate','diagnostic_candidate',deadline,collect=True);val={k:v[256:] for k,v in db.items()};vu=du[256:];commands=hist['commands'];gates=hist['gates'];meancommands=torch.tensor(commands[:256].mean(0),device='cuda');meangates=torch.tensor(gates[:256].mean(0),device='cuda');rng=np.random.default_rng(CFG['donor_seed']);donor=rng.integers(256,size=(256,30));shuffled=torch.tensor(commands[:256][donor,np.arange(30)[None]],device='cuda');command_tests={}
   for name,c in [('fixed',torch.zeros(20,device='cuda')),('stage',meancommands),('shuffle',shuffled)]:
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'command_{name}',deadline,commands=c);command_tests[name]=stats(base[256:]-v)
   holm(command_tests,'Holm3_p');xx=torch.tensor(hist['X'][:256],device='cuda');ss=model.history(xx).reshape(-1,128).double();C=torch.cat([torch.ones(len(ss),1,device='cuda'),ss],-1);logs=torch.tensor(np.maximum(gates[:256],1e-30),device='cuda').reshape(-1,4).double().log();beta=torch.linalg.solve(C.T@C+torch.eye(129,device='cuda'),C.T@logs);np.save(E/'ordinary_router_calibration_beta.npy',beta.cpu().numpy());coord={}
   for name,kwargs in [('ordinary',dict(ordinary_beta=beta)),('uniform',dict(gates=torch.full((30,4),.25,device='cuda'))),('stage',dict(gates=meangates))]:
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'coordination_{name}',deadline,**kwargs);coord[name]=stats(base[256:]-v)
   holm(coord,'Holm3_p');removal={}
   for k in range(4):
    v,_,_=evaluated(model,val,contrasts,vu,'candidate',f'expert{k+1}_removed',deadline,disable=k);removal[f'expert{k+1}']=stats(base[256:]-v)
   holm(removal,'Holm4_p');task=competency(model,hist,deadline);put(E/'SPECIALIZATION_RESULTS.json',dict(competency=task,expert_removal=removal,masking_distribution_shift_caveat=True));put(E/'COMMANDER_RESULTS.json',dict(commands=command_tests,coordination=coord,within_stage_command_SD=commands.std(0).mean(0).tolist(),within_stage_gate_SD=gates.std(0).mean(0).tolist(),ordinary_comparator='calibration-onlyridge1loggateprojectionfromcandidatehistoryencoding;notanEIG-trainedordinaryrouter',stage_and_shuffled_from_calibration_only=True));timing={}
   for method in ['candidate','control','sg']:
    aa,_=actor_load(method);xx=torch.tensor(hist['X'][:1],device='cuda');torch.cuda.synchronize();beg=time.perf_counter()
    for repeat in range(10):
     for t in range(30):aa(xx[:,t])
    torch.cuda.synchronize();timing[method]=dict(milliseconds_per_design=(time.perf_counter()-beg)*1000/300,includes_frozen_belief_likelihood_sensitivity_and_critic=True);del aa
   put(E/'LATENCY.json',timing)
 except Exception as exc:
  import traceback
  traceback.print_exc();status='INCOMPLETE';put(E/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
 finally:
  assert all(sha(O/f'arms/{m}/actor.pt')==h for m,h in endpoints.items());put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-tick,main_policies_completed=list(scores),training_updates=0,endpoints_unchanged=True));print('EVALUATION_END',status,flush=True)
