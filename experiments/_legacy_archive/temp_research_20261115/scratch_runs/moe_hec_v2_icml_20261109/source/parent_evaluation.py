"""Independent terminal prior-EIG bank and frozen full-policy feature interventions."""
import os,sys,time,math,pathlib,json,gc,types
sys.dont_write_bytecode=True
import torch,numpy as np
from torch import nn
from scipy.stats import t as student_t
from native_common import *
from soft_policy import CollaborativeActor,Tasks,prefix_inputs
from option_policy import parts as option_parts
from moe_actor import DirectDesignHeads
from src.research.vendor.vsoed.utils import Net
from frozen_evaluator import contrast_scores
CFG=json.loads((O/'CONFIG_LOCKED.json').read_text());E=O/'evaluation'
def stats(x):
 x=np.asarray(x,float);se=float(x.std(ddof=1)/math.sqrt(len(x)));m=float(x.mean());c=float(student_t.ppf(.975,len(x)-1));return dict(n=len(x),mean=m,MCSE=se,CI95=[m-c*se,m+c*se],p_two_sided=float(2*student_t.sf(abs(m/max(se,1e-300)),len(x)-1)))
def holm(tests,key):
 previous=0
 for rank,name in enumerate(sorted(tests,key=lambda k:tests[k]['p_two_sided'])):previous=max(previous,min(1.,(len(tests)-rank)*tests[name]['p_two_sided']));tests[name][key]=previous

def actor_load(method):
 if method in ['candidate','control']:actor=CollaborativeActor(CFG);path=O/f'arms/{method}/actor.pt'
 else:
  actor=Net([117,256,256,256,2],nn.ReLU,torch.tensor([[-4.,4.]]*2,device='cuda'),'actor');path=O/f'inputs/{method}.pt'
  if method!='vsoed':actor.net=nn.Sequential(*list(actor.net.children())[:4],DirectDesignHeads())
  if method=='reference':actor.net[4].fallback_gate=nn.Linear(256,1)
 actor=actor.cuda();cp=torch.load(path,map_location='cuda',weights_only=False);actor.load_state_dict(cp['actor_state'],strict=True);return actor.eval().requires_grad_(False),path

def native_x(acts,ys,b,device='cuda'):
 t=len(acts);x=torch.zeros(b,117,device=device);x[:,t]=1
 if t:x[:,30:30+2*t]=torch.stack(acts,1).reshape(b,-1).float();x[:,88:88+t]=torch.stack(ys,1).reshape(b,-1).float()
 return x

def simulate(actor,truth,noise,u,method,collect=False,disable=None,stage_gates=None,uniform=False,outside=None):
 n=len(truth);acts=[];ys=[];hist=[];gates=[];features=[];active=None
 for t in range(30):
  x=native_x(acts,ys,n)
  if method in ['candidate','control']:
   a=actor(x,disable=disable,stage_gates=None if stage_gates is None else stage_gates[t],uniform=uniform)
   if collect:
    _,z,g=actor.parts(x);features.append(z);gates.append(g)
  elif method=='reference':
   g,ap=option_parts(actor,x)
   if t%3==0:active=(u[:,t,None]>g.cumsum(-1)).sum(-1).clamp(max=4)
   a=torch.where((active<4)[:,None],ap.gather(1,active.clamp(max=3)[:,None,None].expand(-1,1,2)).squeeze(1),outside(x))
  else:a=actor(x)
  assert bool(((a>=-4)&(a<=4)).all());d=(truth.reshape(n,2,2)-a[:,None]).square().sum(-1);mean=(.1+(d+1e-4).reciprocal().sum(-1)).log();y=mean+.5*noise[:,t];acts.append(a);ys.append(y[:,None]);hist.append(x)
 out=dict(actions=acts,observations=ys,truth=truth)
 if collect:out.update(X=torch.stack(hist,1),features=torch.stack(features,1),gates=torch.stack(gates,1))
 return out

def evaluated(actor,bank,contrast,u,method,label,deadline,collect=False,**kwargs):
 scores=[];hist={};beg=time.monotonic();bench=types.SimpleNamespace(sigma=.5,calls=0)
 for i in range(0,len(bank['truth']),16):
  if time.monotonic()>deadline:raise TimeoutError('Locked evaluation/runtimecap')
  sl=slice(i,i+16);data=simulate(actor,torch.tensor(bank['truth'][sl],device='cuda'),torch.tensor(bank['noise'][sl],device='cuda'),torch.tensor(u[sl],device='cuda'),method,collect=collect,**kwargs);v,_=contrast_scores(bench,data,contrast,16384,'cuda');scores.extend(v[:,-1].numpy())
  if collect:
   entries={k:data[k].cpu().numpy() for k in ['X','features','gates']};entries.update(ds=torch.stack(data['actions'],1).cpu().numpy(),ys=torch.stack(data['observations'],1).cpu().numpy(),truth=data['truth'].cpu().numpy())
   for k,v in entries.items():hist.setdefault(k,[]).append(v)
 scores=np.array(scores);np.save(E/f'{label}_scores.npy',scores);hist={k:np.concatenate(v) for k,v in hist.items()}
 if hist:np.savez_compressed(E/f'{label}_histories.npz',**hist)
 receipt=dict(**stats(scores),seconds=time.monotonic()-beg,simulator_observations=len(scores)*30,contrast_observation_density_calls=bench.calls,positive_observation_density_calls=len(scores)*30);put(E/f'{label}_receipt.json',receipt);print('EVALUATED',label,json.dumps(receipt),flush=True);return scores,hist,receipt

def cka(x,y):
 x=x-x.mean(0);y=y-y.mean(0);cross=x.T@y;return float(np.square(cross).sum()/max(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y),1e-30))

def task_evaluation(hist,deadline):
 from fresh_common import build_fresh
 from native_reward import stateless_rewards
 a=build_fresh('cuda')['algorithm'];post=torch.load(O/'inputs/reference_native_posterior.pt',map_location='cuda',weights_only=False)['posterior']
 for k,nets in a.post_approx.poi_post_nets.items():
  for net,state in zip(nets,post[str(k)]):net.load_state_dict(state);net.eval().requires_grad_(False)
 reward=[]
 for i in range(0,512,16):
  if time.monotonic()>deadline:raise TimeoutError('Taskevaluationcap')
  sl=slice(i,i+16);reward.append(stateless_rewards(a,torch.tensor(hist['ds'][sl],device='cuda',dtype=torch.float32),torch.tensor(hist['ys'][sl],device='cuda',dtype=torch.float32),torch.tensor(hist['truth'][sl],device='cuda',dtype=torch.float32)))
 reward=torch.cat(reward);stages=[3,12,24];row=torch.arange(512,device='cuda').repeat_interleave(3);tt=torch.tensor(stages,device='cuda').repeat(512);ds=torch.tensor(hist['ds'],device='cuda',dtype=torch.float32)[row];ys=torch.tensor(hist['ys'],device='cuda',dtype=torch.float32)[row];th=torch.tensor(hist['truth'],device='cuda',dtype=torch.float32)[row];tasks=Tasks(a,False,CFG['tasks']);labels=tasks.labels(ds,ys,th,tt,reward[row]);targets=[v.cpu().numpy().reshape(512,3,-1) for v in labels];X=torch.tensor(hist['X'][:,stages],device='cuda');metrics={};allfeatures={}
 for method in ['candidate','control','parent']:
  actor,_=actor_load(method);_,z,g=actor.parts(X);allfeatures[method]=z.cpu().numpy();pred=[head(z[:,:,k]) for k,head in enumerate(actor.task_heads)];metrics[method]=dict(within_stage_feature_CKA={f'{k+1}-{l+1}':{str(t):cka(allfeatures[method][:,j,k],allfeatures[method][:,j,l]) for j,t in enumerate(stages)} for k in range(4) for l in range(k+1,4)},gate_means=g.cpu().numpy().mean((0,1)).tolist(),within_stage_gate_SD=g.cpu().numpy().std(0).mean(0).tolist(),heldout_task_MSE_or_Brier=[float(((p.softmax(-1) if k==0 else p)-labels[k].reshape(512,3,-1))[256:].square().mean()) for k,p in enumerate(pred)])
 Z=allfeatures['candidate'];losses=np.zeros((4,4,512,3));probes={}
 for expert in range(4):
  cal=Z[:256,:,expert].reshape(-1,64);mu=cal.mean(0);sd=cal.std(0);sd[sd<1e-6]=1;Fc=np.column_stack([np.ones(len(cal)),(cal-mu)/sd]);Fv=np.column_stack([np.ones(512*3),(Z[:,:,expert].reshape(-1,64)-mu)/sd]);inverse=np.linalg.solve(Fc.T@Fc+np.eye(65),Fc.T)
  for task in range(4):
   beta=inverse@targets[task][:256].reshape(768,-1);prediction=(Fv@beta).reshape(512,3,-1);losses[expert,task]=np.square(prediction-targets[task]).mean(-1);probes[f'expert{expert+1}_task{task+1}']=dict(validation_MSE=float(losses[expert,task,256:].mean()),calibration_MSE=float(losses[expert,task,:256].mean()))
 tests={};rivals=[]
 for task in range(4):
  rivals0=[k for k in range(4) if k!=task];rival=min(rivals0,key=lambda k:losses[k,task,:256].mean());rivals.append(rival);tests[f'task{task+1}_designated_better_than_calibration_rival']=stats((losses[rival,task,256:]-losses[task,task,256:]).mean(1))
 holm(tests,'Holm4_p');success=[k+1 for k,(name,test) in enumerate(tests.items()) if test['mean']>0 and test['Holm4_p']<.05];np.savez_compressed(E/'task_probe_raw.npz',probe_losses=losses,**{f'target{k}':v for k,v in enumerate(targets)})
 return dict(task_head_metrics=metrics,frozen_feature_probe_metrics=probes,designated_expert_tests=tests,calibration_selected_rivals=rivals,credible_distinct_task_experts=success,common_planning_scorer='successfulOption finalfrozenvariationalposterior',planning_estimand='realized discountednative return on candidatejointtrajectories,notexactconditionalEIG',validation_independent_cases=256,stages=stages,deployable_probe_used=False)

def run_evaluation(global_start):
 assert json.loads((O/'evidence/PAIR_COMPLETION.json').read_text())['status']=='COMPLETE';E.mkdir(exist_ok=True);assert not (E/'STARTED.json').exists();beg=time.monotonic();deadline=min(beg+3600,global_start+41400);put(E/'STARTED.json',dict(time=time.time(),pid=os.getpid(),training_updates=0));rng=np.random.default_rng(202611088902);bank=dict(truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)));np.savez_compressed(E/'development_validation_bank.npz',**bank);u=np.random.default_rng(202611088903).random((2000,30));np.save(E/'reference_uniforms.npy',u);contrast=torch.tensor(bank['contrasts'],device='cuda');scores={};receipts={};outside,_=actor_load('original');endpoints={m:sha(O/f'arms/{m}/actor.pt') for m in ['candidate','control']};status='COMPLETE'
 try:
  with torch.no_grad(),torch.device('cuda'):
   for method in ['reference','vsoed','control','candidate']:
    actor,path=actor_load(method);scores[method],_,receipts[method]=evaluated(actor,bank,contrast,u,method,method,deadline,outside=outside);receipts[method]['all_actor_parameters']=sum(p.numel() for p in actor.parameters());receipts[method]['deployment_parameters']=sum(p.numel() for name,p in actor.named_parameters() if not name.startswith('task_heads.'))+(162304 if method=='reference' else 0);del actor
   tests={m:stats(scores['candidate']-scores[m]) for m in ['control','reference']};holm(tests,'Holm_primary_p');tests['vsoed']=stats(scores['candidate']-scores['vsoed']);put(E/'EIG_RESULTS.json',dict(methods=receipts,paired=tests,bank_sha256=sha(E/'development_validation_bank.npz'),finite_contrast_lower_bound=True,compute_matched_native_vsOED=False,primary_family=['candidate-control','candidate-reference']))
   drng=np.random.default_rng(202611089101);db=dict(truth=drng.standard_normal((512,4)),noise=drng.standard_normal((512,30)));du=np.random.default_rng(202611089102).random((512,30));np.savez_compressed(E/'diagnostic_bank.npz',**db,routing_uniforms=du);actor,_=actor_load('candidate');learned,hist,_=evaluated(actor,db,contrast,du,'candidate','diagnostic_learned',deadline,collect=True);val={k:v[256:] for k,v in db.items()};vu=du[256:];stage=torch.tensor(hist['gates'][:256].mean(0),device='cuda');special={}
   for k in range(4):
    v,_,_=evaluated(actor,val,contrast,vu,'candidate',f'disable{k+1}',deadline,disable=k);special[f'learned_minus_disable{k+1}']=stats(learned[256:]-v)
   for name,kw in [('stage_only',dict(stage_gates=stage)),('uniform',dict(uniform=True))]:
    v,_,_=evaluated(actor,val,contrast,vu,'candidate',name,deadline,**kw);special[f'learned_minus_{name}']=stats(learned[256:]-v)
   holm(special,'Holm6_p');g=hist['gates'];Z=hist['features'];functional=dict(gate_mean=g.mean((0,1)).tolist(),within_stage_gate_SD=g.std(0).mean(0).tolist(),stage_explained_variance=(g.mean(0).var(0)/(g.mean(0).var(0)+g.var(0).mean(0)+1e-30)).tolist(),entropy=float(-(g*np.log(np.maximum(g,1e-30))).sum(-1).mean()),within_stage_linear_CKA={f'{k+1}-{l+1}':{str(t):cka(Z[:,t,k],Z[:,t,l]) for t in [3,12,24]} for k in range(4) for l in range(k+1,4)})
   task=task_evaluation(hist,deadline);positive=lambda key:special[key]['mean']>0 and special[key]['Holm6_p']<.05;useful=[k+1 for k in range(4) if positive(f'learned_minus_disable{k+1}')];criterion=len(task['credible_distinct_task_experts'])>=2 and len(useful)>=2 and positive('learned_minus_stage_only');put(E/'SPECIALIZATION_RESULTS.json',dict(functional=functional,paired_interventions=special,task_capabilities=task,credible_useful_experts=useful,specialization_criterion_met=criterion,conditional_exact_EIG='NOT_ESTABLISHED;unresolvedposteriorreference notused'))
   timings={}
   for method in ['candidate','control','reference','vsoed']:
    aa,_=actor_load(method);xx=torch.tensor(hist['X'][:1],device='cuda');torch.cuda.synchronize();t=time.perf_counter()
    for rep in range(100):
     for stage_i in range(30):
      if method=='reference':gp,ap=option_parts(aa,xx[:,stage_i]);zz=outside(xx[:,stage_i])
      else:zz=aa(xx[:,stage_i])
    torch.cuda.synchronize();timings[method]=dict(network_milliseconds_per_design=(time.perf_counter()-t)*1000/3000,reference_includes_proposal_and_fallback=True);del aa
   put(E/'INFERENCE_LATENCY.json',timings)
 except Exception as exc:
  import traceback
  traceback.print_exc();status='INCOMPLETE';put(E/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
 finally:
  assert all(sha(O/f'arms/{m}/actor.pt')==h for m,h in endpoints.items());put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-beg,main_policies_completed=list(scores),endpoints_unchanged=True,training_updates=0));print('EVALUATION_END',status,flush=True)
