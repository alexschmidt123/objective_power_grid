"""Frozen common-bank evaluation,history interventions and competency retention."""
import sys,os,time,json,copy,pathlib,subprocess
sys.dont_write_bytecode=True
import torch,numpy as np
from native_common import *
from soft_policy import CollaborativeActor
from sg_policy import SGActor
from option_policy import parts as option_parts
from src.research.vendor.vsoed.utils import Net
from moe_actor import DirectDesignHeads
from frozen_evaluator import contrast_scores
import parent_evaluation as lib
stats=lib.stats;holm=lib.holm;cka=lib.cka;E=O/'evaluation';CFG=json.loads((O/'CONFIG_LOCKED.json').read_text());PARENT_CONFIG=json.loads((R/'moe_collaborative_soft_queue_20261108/CONFIG_LOCKED.json').read_text());lib.CFG['tasks']=PARENT_CONFIG['tasks'];SOFT=['candidate','control','parent','historical_control']
def actor_load(method):
 if method in ['candidate','control']:
  base=CollaborativeActor(PARENT_CONFIG).cuda();base.load_state_dict(torch.load(O/'inputs/parent.pt',map_location='cuda',weights_only=False)['actor_state']);actor=SGActor(base,copy.deepcopy(base),method=='candidate');path=O/f'arms/{method}/actor.pt'
 elif method in ['parent','historical_control']:actor=CollaborativeActor(PARENT_CONFIG).cuda();path=O/f'inputs/{method}.pt'
 else:
  actor=Net([117,256,256,256,2],torch.nn.ReLU,torch.tensor([[-4.,4.]]*2,device='cuda'),'actor')
  if method!='vsoed':actor.net=torch.nn.Sequential(*list(actor.net.children())[:4],DirectDesignHeads())
  if method=='reference':actor.net[4].fallback_gate=torch.nn.Linear(256,1)
  actor=actor.cuda();path=O/f'inputs/{method}.pt'
 actor.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['actor_state'],strict=True);return actor.eval().requires_grad_(False),path
lib.actor_load=actor_load

def simulate(actor,truth,noise,u,method,collect=False,outside=None,disable=None,routing=None,donor_routes=None,donor_slots=None,replace_expert=None,mechanism=None,parent=None):
 n=len(truth);acts=[];ys=[];record={k:[] for k in ['X','features','gates','retention','anchor','slots']};active=None
 for t in range(30):
  x=lib.native_x(acts,ys,n)
  if method in ['candidate','control']:
   live,anchor,g,r=actor.components(x);over=None
   if routing=='parent':over=(parent.parts(x)[2],r)
   elif routing=='uniform':over=(torch.full_like(g,.25),r)
   elif routing in ['stage','fixed','shuffle']:over=donor_routes[:,t] if routing=='shuffle' else donor_routes[t];over=(over[...,:4],over[...,4:])
   if mechanism=='retention_stage':over=(g,donor_routes[t,4:].expand_as(r))
   a=actor(x,disable=disable,route_override=over,replace_expert=replace_expert,donor_slots=None if donor_slots is None else donor_slots[:,t],mechanism=mechanism)
   if collect:
    first=anchor if actor.anchored else live;delta=live-anchor if actor.anchored else torch.zeros_like(live);items=dict(X=x,features=live,gates=g,retention=r,anchor=anchor,slots=torch.cat([first,r.unsqueeze(-1)*delta],-1))
    for k,v in items.items():record[k].append(v)
  elif method in ['parent','historical_control']:a=actor(x)
  elif method=='reference':
   g,ap=option_parts(actor,x)
   if t%3==0:active=(u[:,t,None]>g.cumsum(-1)).sum(-1).clamp(max=4)
   a=torch.where((active<4)[:,None],ap.gather(1,active.clamp(max=3)[:,None,None].expand(-1,1,2)).squeeze(1),outside(x))
  else:a=actor(x)
  assert bool(((a>=-4)&(a<=4)).all());d=(truth.reshape(n,2,2)-a[:,None]).square().sum(-1);y=(.1+(d+1e-4).reciprocal().sum(-1)).log()+.5*noise[:,t];acts.append(a);ys.append(y[:,None])
 out=dict(actions=acts,observations=ys,truth=truth)
 if collect:out.update({k:torch.stack(v,1) for k,v in record.items()})
 return out

def evaluated(actor,bank,contrast,u,method,label,deadline,collect=False,**kwargs):
 values=[];curves=[];hist={};beg=time.monotonic();bench=type('Benchmark',(),{'sigma':.5,'calls':0})()
 for i in range(0,len(bank['truth']),16):
  if time.monotonic()>deadline:raise TimeoutError('Fixed evaluation budget')
  sl=slice(i,i+16);opts=dict(kwargs)
  for key in ['donor_slots']:
   if opts.get(key) is not None:opts[key]=opts[key][sl]
  if opts.get('routing')=='shuffle':opts['donor_routes']=opts['donor_routes'][sl]
  data=simulate(actor,torch.tensor(bank['truth'][sl],device='cuda'),torch.tensor(bank['noise'][sl],device='cuda'),torch.tensor(u[sl],device='cuda'),method,collect=collect,**opts);score,_=contrast_scores(bench,data,contrast,16384,'cuda');curves.append(score.cpu().numpy());values.extend(score[:,-1].cpu().numpy())
  if collect:
   items={k:data[k].cpu().numpy() for k in ['X','features','gates','retention','anchor','slots']};items.update(ds=torch.stack(data['actions'],1).cpu().numpy(),ys=torch.stack(data['observations'],1).cpu().numpy(),truth=data['truth'].cpu().numpy())
   for k,v in items.items():hist.setdefault(k,[]).append(v)
 values=np.array(values);np.save(E/f'{label}_scores.npy',values);np.save(E/f'{label}_curves.npy',np.concatenate(curves));hist={k:np.concatenate(v) for k,v in hist.items()}
 if hist:np.savez_compressed(E/f'{label}_histories.npz',**hist)
 receipt=dict(**stats(values),seconds=time.monotonic()-beg,simulator_observations=len(values)*30,negative_density_calls=bench.calls,positive_density_calls=len(values)*30);put(E/f'{label}_receipt.json',receipt);print('EVALUATED',label,json.dumps(receipt),flush=True);return values,hist,receipt

def probe_retention(hist,task):
 raw=np.load(E/'task_probe_raw.npz');targets=[raw[f'target{k}'] for k in range(4)];X=torch.tensor(hist['X'][:,[3,12,24]],device='cuda');data={};anchor_exact=True
 parent,_=actor_load('parent');pz=parent.parts(X)[1].cpu().numpy()
 for method in ['candidate','control','parent']:
  actor,_=actor_load(method)
  if method=='parent':data[method]=pz;continue
  live,anchor,g,r=actor.components(X);anchor=anchor.cpu().numpy();anchor_exact=anchor_exact and np.array_equal(anchor,pz);data[method]=live.cpu().numpy();data[method+'_joint']=torch.cat([torch.tensor(anchor,device='cuda'),r.unsqueeze(-1)*(live-torch.tensor(anchor,device='cuda'))],-1).cpu().numpy()
 assert anchor_exact,'Anchor feature preservation requirement'
 data['parent_joint']=np.concatenate([pz,np.zeros_like(pz)],-1);losses={}
 for method,Z in data.items():
  scores=np.zeros((4,512,3))
  for k in range(4):
   d=Z.shape[-1];cal=Z[:256,:,k].reshape(-1,d).astype(float);mu=cal.mean(0);sd=cal.std(0);sd[sd<1e-6]=1;C=np.column_stack([np.ones(len(cal)),(cal-mu)/sd]);F=np.column_stack([np.ones(1536),(Z[:,:,k].reshape(-1,d)-mu)/sd]);beta=np.linalg.solve(C.T@C+np.eye(d+1),C.T@targets[k][:256].reshape(768,-1));prediction=(F@beta).reshape(512,3,-1);scores[k]=np.square(prediction-targets[k]).mean(-1)
  losses[method]=scores
 tests={f'expert{k+1}_live_improvement_vs_parent':stats((losses['parent'][k,256:]-losses['candidate'][k,256:]).mean(1)) for k in range(4)};holm(tests,'Holm4_p');np.savez_compressed(E/'competency_retention_raw.npz',**losses);return dict(anchor_features_bitwise_parent_equal=True,live_expert_improvement_tests=tests,validation_MSE={m:v[:,256:].mean((1,2)).tolist() for m,v in losses.items()},joint128_vs_parent64_zero_padding_capacity_caveat=True)

def run_evaluation(start):
 assert json.loads((O/'evidence/PAIR_COMPLETION.json').read_text())['status']=='COMPLETE';E.mkdir(exist_ok=True);assert not (E/'STARTED.json').exists();beg=time.monotonic();deadline=min(start+10800,beg+2400);put(E/'STARTED.json',dict(time=time.time(),training_updates=0));rng=np.random.default_rng(2026110813902);bank=dict(truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)));np.savez_compressed(E/'development_validation_bank.npz',**bank);u=np.random.default_rng(2026110813903).random((2000,30));np.save(E/'routing_uniforms.npy',u);contrast=torch.tensor(bank['contrasts'],device='cuda');scores={};methods={};status='COMPLETE';outside,_=actor_load('original');endpoints={m:sha(O/f'arms/{m}/actor.pt') for m in ['candidate','control']}
 try:
  with torch.no_grad(),torch.device('cuda'):
   for method in ['parent','reference','vsoed','ordinary','historical_control','control','candidate']:
    actor,path=actor_load(method);scores[method],_,methods[method]=evaluated(actor,bank,contrast,u,method,method,deadline,outside=outside);methods[method]['actor_parameters']=sum(p.numel() for p in actor.parameters());methods[method]['frozen_anchor_weights']=sum(v.numel() for name,v in actor.named_buffers() if name.startswith(('anchor_encoder.','anchor_experts.')));methods[method]['deployed_network_parameters_and_weights']=sum(p.numel() for name,p in actor.named_parameters() if not name.startswith('task_heads.'))+methods[method]['frozen_anchor_weights']+(162304 if method=='reference' else 0);del actor
   for method in ['dad','rl_sboed','scorebed']:
    remaining=deadline-time.monotonic();assert remaining>0
    with (E/f'{method}_worker.log').open('wb') as f:result=subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(O/'source/historical_worker.py'),method],stdout=f,stderr=subprocess.STDOUT,timeout=remaining,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
    if result.returncode:raise RuntimeError(f'{method} frozen baseline evaluation failed;see workerlog')
    methods[method]=json.loads((E/f'{method}_receipt.json').read_text());scores[method]=np.load(E/f'{method}_scores.npy')
   primary=stats(scores['candidate']-scores['control']);secondary={m:stats(scores['candidate']-scores[m]) for m in ['parent','reference','vsoed','historical_control']};holm(secondary,'Holm4_p');other={m:stats(scores['candidate']-scores[m]) for m in ['ordinary','dad','rl_sboed','scorebed']};put(E/'EIG_RESULTS.json',dict(methods=methods,primary_candidate_minus_control=primary,secondary=secondary,other_comparisons_descriptive=other,bank_sha256=sha(E/'development_validation_bank.npz'),negative_prior_contrasts=1000000,total_denominator_terms=1000001,finite_contrast_lower_bound=True,not_compute_matched_against_frozen_references=True))
   drng=np.random.default_rng(2026110814101);db=dict(truth=drng.standard_normal((512,4)),noise=drng.standard_normal((512,30)));du=drng.random((512,30));np.savez_compressed(E/'diagnostic_bank.npz',**db,routing_uniforms=du);actor,_=actor_load('candidate');parent,_=actor_load('parent');base,hist,_=evaluated(actor,db,contrast,du,'candidate','diagnostic_candidate',deadline,collect=True);val={k:v[256:] for k,v in db.items()};vu=du[256:];calroutes=np.concatenate([hist['gates'][:256],hist['retention'][:256]],-1);stage=torch.tensor(calroutes.mean(0),device='cuda');fixed=stage.mean(0).expand(30,8);rrng=np.random.default_rng(2026110814102);donor=rrng.integers(256,size=(256,30));jj=np.arange(30)[None];shuffled=torch.tensor(calroutes[donor,jj],device='cuda');slots=torch.tensor(hist['slots'][:256][donor,jj],device='cuda');route_tests={}
   for name,opts in [('parent',dict(routing='parent',parent=parent)),('uniform',dict(routing='uniform')),('fixed',dict(routing='fixed',donor_routes=fixed)),('stage',dict(routing='stage',donor_routes=stage)),('shuffle',dict(routing='shuffle',donor_routes=shuffled))]:
    v,_,_=evaluated(actor,val,contrast,vu,'candidate',f'route_{name}',deadline,**opts);route_tests[name]=stats(base[256:]-v)
   holm(route_tests,'Holm5_p');donor_tests={};removal={}
   for k in range(4):
    v,_,_=evaluated(actor,val,contrast,vu,'candidate',f'expert{k+1}_donor',deadline,replace_expert=k,donor_slots=slots);donor_tests[f'expert{k+1}']=stats(base[256:]-v)
    v,_,_=evaluated(actor,val,contrast,vu,'candidate',f'expert{k+1}_zero',deadline,disable=k);removal[f'expert{k+1}']=stats(base[256:]-v)
   holm(donor_tests,'Holm4_p');holm(removal,'Holm4_p');mechanism={}
   for name,opts in [('adaptive_off',dict(mechanism='adaptive_off')),('retention_stage',dict(mechanism='retention_stage',donor_routes=stage)),('anchors_live',dict(mechanism='anchors_live'))]:
    v,_,_=evaluated(actor,val,contrast,vu,'candidate',f'mechanism_{name}',deadline,**opts);mechanism[name]=stats(base[256:]-v)
   holm(mechanism,'Holm3_p');task=lib.task_evaluation(hist,deadline);retain=probe_retention(hist,task);g=hist['gates'];r=hist['retention'];features=hist['features'];positive=lambda table,k:table[k]['mean']>0 and next((v for key,v in table[k].items() if key.startswith('Holm')),1)<.05
   special=dict(task_capabilities=task,competency_retention=retain,donor_expert_tests=donor_tests,zero_removal_sensitivity=removal,masking_joint_distribution_shift_caveat=True,all_experts_useful_by_donor_test=all(positive(donor_tests,k) for k in donor_tests),functional=dict(gate_means=g.mean((0,1)).tolist(),retention_means=r.mean((0,1)).tolist(),within_stage_gate_SD=g.std(0).mean(0).tolist(),within_stage_retention_SD=r.std(0).mean(0).tolist(),within_stage_CKA={f'{k+1}-{l+1}':{str(t):cka(features[:,t,k],features[:,t,l]) for t in [3,12,24]} for k in range(4) for l in range(k+1,4)}));put(E/'SPECIALIZATION_RESULTS.json',special);put(E/'ROUTING_RESULTS.json',dict(tests=route_tests,history_routing_beyond_stage=positive(route_tests,'stage') and positive(route_tests,'fixed'),donor_sampling='independentstage-matchedcalibrationhistories'));put(E/'MECHANISM_RESULTS.json',dict(tests=mechanism,ablation_distribution_shift_caveat=True,effective_capacity_differs_from_exact_ordinary_control=True))
   from conditional_diagnostics import run_conditional
   run_conditional(actor,hist,deadline)
   timings={}
   for method in ['candidate','control','parent','reference','vsoed']:
    aa,_=actor_load(method);xx=torch.tensor(hist['X'][:1],device='cuda');torch.cuda.synchronize();tick=time.perf_counter()
    for repeat in range(100):
     for t in range(30):
      if method=='reference':gp,ap=option_parts(aa,xx[:,t]);aout=outside(xx[:,t])
      else:aout=aa(xx[:,t])
    torch.cuda.synchronize();timings[method]=dict(milliseconds_per_design=(time.perf_counter()-tick)*1000/3000,definition='allactualfeature/anchorforwards;referenceincludesfallback/proposals');del aa
   put(E/'INFERENCE_LATENCY.json',timings)
 except Exception as exc:
  import traceback
  traceback.print_exc();status='INCOMPLETE';put(E/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
 finally:
  assert all(sha(O/f'arms/{m}/actor.pt')==h for m,h in endpoints.items());put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-beg,main_policies_completed=list(scores),training_updates=0,endpoints_unchanged=True));print('EVALUATION_END',status,flush=True)
