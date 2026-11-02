"""Revised isolated SIR improvement: max12additional training hours, no preemption."""
import time
ENTRY=time.monotonic()
import os,sys,json,csv,copy,math,hashlib,subprocess,fcntl,traceback,io,unittest.mock,signal
from pathlib import Path
import torch,numpy as np,cloudpickle
from src.research.sir_t3_campaign import write,sha
from src.research.sir_t3_common import T3SIR,fixed_shared_spce
from src.research.sir_t3_improvement_numerics import CountSortedSIR,schedule_policy,physical_times
from src.research.engine import rollout,generator
from src.research.sir_t3_evaluation import evaluate,stats
R=Path(__file__).resolve().parents[2];C=R/'scratch_runs/sir_ode_t3_baseline_improvement';cfg=json.loads((C/'configs/protocol.json').read_text());P=Path(cfg['parent']);O=Path(cfg['original_extension'])
def freeze_check():
 for rel,h in json.loads((C/'manifests/source_sha256.json').read_text()).items():assert sha(R/rel)==h,'Source changed: '+rel
 for path,h in json.loads((C/'manifests/original_preservation_sha256.json').read_text()).items():assert sha(path)==h,'Original artifact changed: '+path

def leases():
 ids=[int(v) for v in os.environ.get('SIR_IMPROVE_LOCK_FDS','').split(',') if v];assert len(ids)==3,'Worker requires exclusive GPU lease';return [os.fdopen(os.dup(v),'a+') for v in ids]
class Budget:
 def __init__(self,method):
  self.method=method;self.p=C/'manifests/aggregate_budget.json';x=json.loads(self.p.read_text());assert x['active'] is None,'Review interrupted stage before recovery';self.prior=x['spent'][method];self.start=ENTRY;x['active']=dict(method=method,start=ENTRY,prior=self.prior,PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));write(self.p,x);self.bound=2.
  def cutoff(*args):raise TimeoutError('Additional training budget deadline; retain existing checkpoints')
  signal.signal(signal.SIGALRM,cutoff);signal.setitimer(signal.ITIMER_REAL,max(.01,self.left()-30))
 def spent(self):return self.prior+time.monotonic()-self.start
 def left(self):
  x=json.loads(self.p.read_text());return min(x['caps'][self.method]-self.spent(),x['cap_seconds']-sum(v for k,v in x['spent'].items() if k!=self.method)-self.spent())
 def can(self,bound=None,reserve=30):return self.left()>max(reserve,1.5*(self.bound if bound is None else bound)+5)
 def finish(self,state):
  signal.setitimer(signal.ITIMER_REAL,0);x=json.loads(self.p.read_text());x['spent'][self.method]=self.spent();x['active']=None;x.setdefault('history',[]).append(dict(method=self.method,status=state,charged_seconds=self.spent(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));assert x['spent'][self.method]<=x['caps'][self.method] and sum(x['spent'].values())<=x['cap_seconds'],'Budget exceeded';write(self.p,x);return x['spent'][self.method]
 def status(self,state,**kw):
  x=dict(method=self.method,state=state,PID=os.getpid(),charged_training_seconds=self.spent(),remaining_seconds=self.left(),**kw);write(C/'manifests'/f'{self.method}_status.json',x);print(json.dumps(x),flush=True)

def checkpoint(method,p,opt,sch,u,budget,extra=None):
 d=C/'checkpoints'/method;d.mkdir(exist_ok=True);path=d/f'update{u}_{int(time.time())}.pt';tmp=path.with_suffix('.partial');torch.save(dict(policy=copy.deepcopy(p).cpu(),optimizer=opt.state_dict() if opt else None,scheduler=sch.state_dict() if sch else None,update=u,configuration=cfg,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),source_hash=sha(C/'manifests/source_sha256.json'),extra=extra),tmp);tmp.replace(path);row=dict(path=str(path),sha256=sha(path),update=u,charged_training_seconds=budget.spent(),eligible=budget.left()>=0);write(path.with_suffix('.json'),row)
 if row['eligible']:write(d/'latest.json',row)
 return row

def development(method,p,update,budget):
 out=C/'evaluations/development';out.mkdir(exist_ok=True)
 with torch.no_grad():values,meta=evaluate(cfg['benchmark'],p,cfg['seeds']['development'],128,[65536],out,method+'_update'+str(update))
 path=C/'logs'/(method+'_development.json');rows=json.loads(path.read_text()) if path.exists() else [];rows.append(dict(update=update,charged_seconds=budget.spent(),EIG=stats(values[65536]),metadata=meta));write(path,rows)


def fixed():
 lease=leases();freeze_check();budget=Budget('fixed');torch.set_num_threads(4)
 try:
  with torch.device('cuda:0'):
   b=T3SIR(cfg['benchmark']);g=torch.Generator(device='cpu').manual_seed(cfg['seeds']['screen']);U=torch.rand(64,3,generator=g,dtype=torch.float64,device='cpu').sort(-1).values;times=.1*torch.arange(1,4,dtype=torch.float64,device='cpu')+(100-.3)*U
   # Half the starts cover early/medium schedules; all lie in the same feasible space.
   times[32:]=.1*torch.arange(1,4,dtype=torch.float64,device='cpu')+(40-.3)*U[32:];times[0]=torch.tensor(json.loads((C/'audits/fixed_cpu_correctness.json').read_text())['old_fixed_times'],device='cpu');screen=[]
   for i,t in enumerate(times):
    p=schedule_policy(b,t).cuda()
    with torch.no_grad():v=fixed_shared_spce(b,p,128,16384,cfg['seeds']['screen'])
    screen.append(dict(index=i,times=t.tolist(),mean=float(v.mean())))
   ranking=sorted(screen,key=lambda x:-x['mean']);chosen=[]
   for row in ranking:
    if not chosen or min(np.linalg.norm((np.asarray(row['times'])-np.asarray(z['times']))/100) for z in chosen)>.05:chosen.append(row)
    if len(chosen)==8:break
   if len(chosen)<8:chosen+=(z for z in ranking if z not in chosen)
   chosen=chosen[:8];write(C/'audits/fixed_screening.json',dict(candidates=screen,selected=chosen,CRN=True,full_horizon=True,seed=cfg['seeds']['screen']));models=[schedule_policy(b,z['times']).cuda() for z in chosen];opts=[torch.optim.Adam(p.parameters(),lr=.005,betas=(.8,.998)) for p in models];sched=[torch.optim.lr_scheduler.ExponentialLR(o,.98) for o in opts];history=[];u=0;last_save=budget.spent();maxround=5.
   for u in range(1,2001):
    if not budget.can(maxround,reserve=180) or budget.spent()+maxround>=6000:u-=1;break
    t=time.monotonic();losses=[]
    for i,p in enumerate(models):
     opts[i].zero_grad();v=fixed_shared_spce(b,p,512,16384,cfg['seeds']['refinement']+u*104729);loss=-v.mean();assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None);opts[i].step()
     if u%250==0:sched[i].step()
     losses.append(float(loss.detach()))
    torch.cuda.synchronize();maxround=max(maxround,time.monotonic()-t)
    if u==1 or u%25==0:
     row=dict(round=u,losses=losses,times=[physical_times(b,p) for p in models],charged_seconds=budget.spent());history.append(row);write(C/'logs/fixed_learning.json',history);budget.status('MULTISTART_REFINEMENT',optimizer_updates_per_start=u,total_optimizer_updates=u*8)
    if u==1 or budget.spent()-last_save>=300:
     for i,p in enumerate(models):checkpoint('fixed_start'+str(i),p,opts[i],sched[i],u,budget)
     last_save=budget.spent()
   # Independent development selection is part of charged optimization.
   values=[]
   for i,p in enumerate(models):
    with torch.no_grad():v=fixed_shared_spce(b,p,512,65536,cfg['fixed']['selection_seed'])
    np.save(C/'evaluations'/f'fixed_development_start{i}.npy',v.cpu().numpy());values.append(float(v.mean()))
   best=int(np.argmax(values));cp=checkpoint('fixed_corrected',models[best],opts[best],sched[best],u,budget,extra=dict(selected_start=best,selection_development_means=values,selection_seed=cfg['fixed']['selection_seed'],times=physical_times(b,models[best])));total=budget.finish('MULTISTART_PROTOCOL_COMPLETE');result=dict(method='fixed',variant='corrected_multistart_joint_spce',training_seed=101,training_objective='joint full-horizon sPCE',batch_size=512,training_contrasts=16384,optimizer_updates=u*8,selected_optimizer_updates=u,fresh_training_trajectories=u*8*512+64*128+8*512,charged_training_seconds=total,additional_training_seconds=total,native_schedule_complete=False,convergence_status='COMPUTE_LIMITED',checkpoint=cp,times=physical_times(b,models[best]),selection_development_means=values,source_hash=sha(C/'manifests/source_sha256.json'));write(C/'manifests/fixed_complete.json',result);(C/'reports/03_FIXED_OPTIMIZATION.md').write_text('# Corrected Fixed result\n\n'+json.dumps(result,indent=2))
 except Exception as e:budget.finish('FAILED');write(C/'manifests/fixed_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

def dad():
 lease=leases();freeze_check();budget=Budget('dad');torch.set_num_threads(4)
 try:
  original=O/'manifests/dad_complete.json'
  if not original.exists():budget.finish('BLOCKED_NO_ORIGINAL_COMPLETED_DAD');write(C/'manifests/dad_blocked.json',dict(reason='Original completed DAD checkpoint unavailable'));return
  old=json.loads(original.read_text());cpold=old['checkpoint'];assert sha(cpold['path'])==cpold['sha256'];saved=torch.load(cpold['path'],map_location='cpu',weights_only=False)
  with torch.device('cuda:0'):
   # Independent physical/input sensitivities, including high-rate and boundary cases.
   g=generator(cfg['seeds']['numerics']);theta=torch.exp(torch.randn(257,2,generator=g,dtype=torch.float64)*.5+torch.tensor([math.log(.5),math.log(.1)]));theta[-1]=torch.tensor([7.,.01]);action=torch.linspace(.10001,99.9,257,dtype=torch.float64)[:,None];weight=torch.randn(257,4,generator=g,dtype=torch.float64);checks=[]
   for cls in [T3SIR,CountSortedSIR]:
    b=cls(cfg['benchmark']);th=theta.clone().requires_grad_();state=b.initial((257,)).requires_grad_();a=action.clone().requires_grad_();y,out=b.forward(th,state,a);grads=torch.autograd.grad((out*weight).sum(),(th,state,a));checks.append([y.detach(),out.detach(),*[z.detach() for z in grads]])
   for a,bv in zip(*checks):torch.testing.assert_close(a,bv,rtol=1e-8,atol=1e-7)
   rows=[];policy_grads=[];infos=[]
   for name,cls in [('reference',T3SIR),('count_sorted',CountSortedSIR)]:
    for rep in range(2):
     p=copy.deepcopy(saved['policy']).cuda().train();b=cls(cfg['benchmark']);p.zero_grad();torch.cuda.synchronize();t=time.monotonic();r=rollout(b,p,3,2000,cfg['seeds']['numerics']+1,2000);torch.cuda.synchronize();forward=time.monotonic()-t;info=r['info'][:,-1];loss=-info.mean();t=time.monotonic();loss.backward();torch.cuda.synchronize();back=time.monotonic()-t;pg=torch.cat([x.grad.flatten() for x in p.parameters()]).detach().cpu();rows.append(dict(implementation=name,repeat=rep,forward_seconds=forward,backward_seconds=back,total_seconds=forward+back,training_sPCE=float(info.mean().detach()),saturation_fraction=float((info>math.log(2001)-.1).double().mean())));policy_grads.append(pg);infos.append(info.detach().cpu());del r,loss,info,p
   torch.testing.assert_close(infos[1],infos[3],rtol=1e-8,atol=1e-7);torch.testing.assert_close(policy_grads[1],policy_grads[3],rtol=1e-7,atol=1e-6);speedup=rows[1]['total_seconds']/rows[3]['total_seconds'];validation=dict(status='PASS',physical_predictions=True,loglikelihood_and_terminal_sPCE=True,time_parameter_initial_state_gradients=True,policy_gradients=True,high_rate_boundary_cases=True,records=rows,speedup=speedup,sort_only_no_physics_change=True);write(C/'audits/dad_gpu_validation.json',validation);(C/'reports/04_DAD_RUNTIME_PROFILE.md').write_text('# Native DAD profile\n\n'+json.dumps(validation,indent=2));(C/'reports/05_DAD_NUMERICAL_VALIDATION.md').write_text('# Numerical/gradient validation\n\n'+json.dumps(validation,indent=2))
   if speedup<1.05:total=budget.finish('BLOCKED_NO_DEMONSTRATED_ACCELERATION');write(C/'manifests/dad_blocked.json',dict(reason='Measured speedup below1.05; no knowingly ineffective extension launched',speedup=speedup,charged_seconds=total));return
   p=saved['policy'].cuda();opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));opt.load_state_dict(saved['optimizer']);scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,.98);scheduler.load_state_dict(saved['scheduler']);b=CountSortedSIR(cfg['benchmark']);startu=saved['update'];u=startu;nextcp=budget.spent()+300;native_unit=rows[3]['total_seconds'];history=[];nextdev=3600;development('dad',p,u,budget)
   while u<50000 and budget.can(native_unit,reserve=120):
    u+=1;t=time.monotonic();opt.zero_grad();r=rollout(b,p,3,2000,101+u*104729,2000);info=r['info'][:,-1];loss=-info.mean();assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None);opt.step()
    if (u-1)%1000==0:scheduler.step()
    torch.cuda.synchronize();native_unit=max(native_unit,time.monotonic()-t)
    if u==startu+1 or u%25==0:
     row=dict(update=u,new_updates=u-startu,charged_seconds=budget.spent(),training_sPCE=float(info.mean().detach()),saturation_fraction=float((info>math.log(2001)-.1).double().mean()),lr=opt.param_groups[0]['lr']);history.append(row);write(C/'logs/dad_learning.json',history);budget.status('ACCELERATED_NATIVE_DAD_TRAINING',optimizer_updates=u,additional_updates=u-startu)
    if u==startu+1 or budget.spent()>=nextcp:checkpoint('dad_accelerated',p,opt,scheduler,u,budget,extra=dict(parent_checkpoint=cpold,additional_updates=u-startu));nextcp=budget.spent()+300
    if budget.spent()>=nextdev and budget.can(native_unit,reserve=180):development('dad',p,u,budget);nextdev+=3600
   cp=checkpoint('dad_accelerated',p,opt,scheduler,u,budget,extra=dict(parent_checkpoint=cpold,additional_updates=u-startu));total=budget.finish('COMPLETE');result=dict(method='dad',variant='count_sorted_exact_native_continuation',training_seed=101,training_objective='full-horizon native sPCE',batch_size=2000,training_contrasts=2000,optimizer_updates=u,additional_optimizer_updates=u-startu,fresh_training_trajectories=(u-startu)*2000,charged_training_seconds=old['charged_training']['charged_seconds']+total,additional_training_seconds=total,native_schedule_complete=u>=50000,convergence_status='NATIVE_SCHEDULE_COMPLETE' if u>=50000 else 'COMPUTE_LIMITED',checkpoint=cp,source_hash=sha(C/'manifests/source_sha256.json'));write(C/'manifests/dad_complete.json',result);(C/'reports/06_DAD_TRAINING_SUFFICIENCY.md').write_text('# DAD continuation\n\n'+json.dumps(result,indent=2))
 except Exception as e:budget.finish('FAILED');write(C/'manifests/dad_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

def load_runtime(path):
 with unittest.mock.patch('torch.storage._load_from_bytes',lambda data:torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)):
  with open(path,'rb') as f:return cloudpickle.load(f)

def rl():
 lease=leases();freeze_check();budget=Budget('rl_sboed');torch.set_num_threads(4)
 try:
  old=json.loads((P/'manifests/rl_sboed_complete.json').read_text());cpold=old['checkpoint'];assert sha(cpold['native_state'])==cpold['native_state_sha256'];runtime=load_runtime(cpold['native_state']);saved=torch.load(cpold['path'],map_location='cpu',weights_only=False);p=runtime['policy'];a=runtime['algorithm'];replay=runtime['replay'];assert a.policy is p.native;assert all(torch.equal(v,saved['policy'].state_dict()[k]) for k,v in p.state_dict().items());opts=[a._policy_optimizer,*a._qf_optimizers,a._alpha_optimizer];alpha_before=float(a._log_alpha.detach());alpha_step=float(next(iter(a._alpha_optimizer.state.values()))['step']);seen=set()
  # Preserve parameter identities and alpha optimizer; SAC.to() would reset them.
  for net in a.networks:
   for q in net.parameters():
    if id(q) in seen:continue
    seen.add(id(q));q.data=q.data.cuda()
    if q.grad is not None:q.grad=q.grad.cuda()
   for q in net.buffers():q.data=q.data.cuda()
  a._log_alpha.data=a._log_alpha.data.cuda()
  if a._log_alpha.grad is not None:a._log_alpha.grad=a._log_alpha.grad.cuda()
  for opt in opts:
   for state in opt.state.values():
    for k,v in state.items():
     if torch.is_tensor(v):state[k]=v.cpu() if k=='step' else v.cuda()
  assert a._alpha_optimizer.param_groups[0]['params'][0] is a._log_alpha and float(a._log_alpha.detach())==alpha_before;assert float(next(iter(a._alpha_optimizer.state.values()))['step'])==alpha_step;replay.data=replay.data.cuda();np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']]);u=saved['update'];startu=u;episodes=saved['episodes'];gradients=old['optimizer_updates'];nextcp=budget.spent()+300;unit=10.;history=[];nextdev=3600
  with torch.device('cuda:0'):
   b=T3SIR(cfg['benchmark']);development('rl_sboed',p,u,budget)
   while budget.can(unit,reserve=120):
    t=time.monotonic();u+=1
    with torch.no_grad():r=rollout(b,p,3,100,101+u*104729,100000)
    info=r['info'];reward=info-torch.cat((torch.zeros_like(info[:,:1]),info[:,:-1]),1);inputs=torch.stack(r['features'],1).reshape(300,-1).float();nxt=torch.stack(r['features'][1:]+[r['final_features']],1).reshape_as(inputs).float();actions=torch.stack(r['actions'],1)[...,0];previous=torch.cat((actions.new_zeros(100,1),actions[:,:-1]),1);lo=previous+b.gap;hi=b.end-(3-torch.arange(3,device=actions.device)-1)*b.gap;unitaction=(2*(actions-lo)/(hi-lo)-1).reshape(-1,1).float();done=torch.zeros(100,3);done[:,-1]=1;replay.add(torch.cat((inputs,unitaction,reward.reshape(-1,1).float(),done.reshape(-1,1),nxt),-1));episodes+=100
    for _ in range(64):
     aloss,qloss,entropy=a.optimize_policy(replay.sample(4096,3));a._update_targets();gradients+=1;assert all(torch.isfinite(z).all() for z in [aloss,*qloss,entropy,a._log_alpha])
    a._target_entropy-=a._ent_anneal_rate;torch.cuda.synchronize();unit=max(unit,time.monotonic()-t)
    if u==startu+1 or u%25==0:
     row=dict(collection_iteration=u,gradient_updates=gradients,additional_seconds=budget.spent(),cumulative_seconds=old['charged_training']['charged_seconds']+budget.spent(),training_sPCE=float(info[:,-1].mean()),replay_count=replay.count,alpha=float(a._log_alpha.exp().detach()));history.append(row);write(C/'logs/rl_learning.json',history);budget.status('EXACT_NATIVE_RL_CONTINUATION',optimizer_updates=gradients,collection_iteration=u)
    if u==startu+1 or budget.spent()>=nextcp:
     cp=checkpoint('rl_sboed_extended',p,a._policy_optimizer,None,u,budget,extra=dict(actor_optimizer_updates=gradients,episodes=episodes,parent_checkpoint=cpold));native=Path(cp['path']).with_suffix('.native.pkl')
     with native.open('wb') as f:cloudpickle.dump(runtime,f)
     write(native.with_suffix('.json'),dict(sha256=sha(native),actor_optimizer_updates=gradients,collection_iteration=u,episodes=episodes));nextcp=budget.spent()+300
    if budget.spent()>=nextdev and budget.can(unit,reserve=180):development('rl_sboed',p,u,budget);nextdev+=3600
   cp=checkpoint('rl_sboed_extended',p,a._policy_optimizer,None,u,budget,extra=dict(actor_optimizer_updates=gradients,episodes=episodes,parent_checkpoint=cpold));native=Path(cp['path']).with_suffix('.native.pkl')
   with native.open('wb') as f:cloudpickle.dump(runtime,f)
   cp.update(native_state=str(native),native_state_sha256=sha(native));total=budget.finish('COMPLETE');result=dict(method='rl_sboed',variant='exact_native_replay_continuation_6h',training_seed=101,training_objective='native SAC/REDQ full-horizon information increments',batch_size=100,training_contrasts=100000,optimizer_updates=gradients,additional_optimizer_updates=gradients-old['optimizer_updates'],fresh_training_trajectories=episodes-saved['episodes'],charged_training_seconds=old['charged_training']['charged_seconds']+total,additional_training_seconds=total,native_schedule_complete=False,convergence_status='COMPUTE_LIMITED',checkpoint=cp,source_hash=sha(C/'manifests/source_sha256.json'),full_state_resume=True);write(C/'manifests/rl_sboed_complete.json',result);(C/'reports/07_RL_SBOED_EXTENSION.md').write_text('# Exact native continuation\n\n'+json.dumps(result,indent=2))
 except Exception as e:budget.finish('FAILED');write(C/'manifests/rl_sboed_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

def final_evaluation():
 lease=leases();freeze_check();torch.set_num_threads(4);records={};models={'random':None};sources={'fixed':O,'dad':O,'rl_sboed':P,'vsoed':P}
 for m,owner in sources.items():
  new=C/'manifests'/f'{m}_complete.json';old=owner/'manifests'/f'{m}_complete.json';path=new if new.exists() else old
  if not path.exists():continue
  r=json.loads(path.read_text());cp=r['checkpoint'];assert sha(cp['path'])==cp['sha256'];records[m]=r;models[m]=torch.load(cp['path'],map_location='cpu',weights_only=False)['policy']
 for m,owner in sources.items():
  if (C/'manifests'/f'{m}_complete.json').exists():
   oldpath=owner/'manifests'/f'{m}_complete.json'
   if oldpath.exists():
    old=json.loads(oldpath.read_text());cp=old['checkpoint'];assert sha(cp['path'])==cp['sha256'];tag=m+'_original_4h';models[tag]=torch.load(cp['path'],map_location='cpu',weights_only=False)['policy'];records[tag]=old
 # Freeze ALL candidate identifiers before generating/accessing fresh confirmation arrays.
 write(C/'manifests/final_candidates_frozen.json',{m:r['checkpoint'] for m,r in records.items()});start=time.monotonic();scores={};result={};out=C/'evaluations/fresh_confirmation';out.mkdir(exist_ok=False)
 resolution={}
 for m,p in models.items():
  vals,meta=evaluate(cfg['benchmark'],p,cfg['seeds']['development'],128,[262144,1000000],C/'evaluations/resolution',m,method='random' if m=='random' else None);resolution[m]=dict(change_262k_minus_1M=stats(vals[262144]-vals[1000000]),metadata=meta)
 write(C/'audits/evaluation_resolution.json',dict(results=resolution,final_common_contrasts=1000000,not_method_specific=True))
 for m,p in models.items():
  values,meta=evaluate(cfg['benchmark'],p,cfg['seeds']['confirmation'],2048,[1000000],out,m,method='random' if m=='random' else None);scores[m]=values[1000000];result[m]=dict(**stats(scores[m]),metadata=meta);write(C/'evaluations/final_results.json',result)
 paired={a+'-minus-'+b:stats(scores[a]-scores[b]) for a in scores for b in ['random','dad','rl_sboed','vsoed'] if b in scores and a!=b}
 for m in ['fixed','dad','rl_sboed']:
  if m+'_original_4h' in scores:paired[m+'-minus-original4h']=stats(scores[m]-scores[m+'_original_4h'])
 write(C/'evaluations/paired_comparisons.json',paired);write(C/'manifests/final_evaluation_cost.json',dict(seconds=time.monotonic()-start,training=False,separate_from_additional_training_budget=True));rows=[]
 for m in ['random','fixed','dad','rl_sboed','vsoed','scorebed']:
  r=records.get(m,{});s=result.get(m,{});variant=r.get('variant','original_fourhour_reference' if m!='scorebed' else 'SCOREBED_NATIVE_BUDGET_INFEASIBLE');rows.append(dict(method=m,variant=variant,training_seed=101,training_objective=r.get('training_objective','no training' if m=='random' else 'native original'),batch_size=r.get('batch_size',''),training_contrasts=r.get('training_contrasts',r.get('contrasts','')),optimizer_updates=r.get('optimizer_updates',0),fresh_training_trajectories=r.get('fresh_training_trajectories',r.get('generated_training_trajectories',0)),charged_training_seconds=r.get('charged_training_seconds',r.get('charged_training',{}).get('charged_seconds',0)),additional_training_seconds=r.get('additional_training_seconds',0),native_schedule_complete=r.get('native_schedule_complete',m=='random'),convergence_status=r.get('convergence_status','PLATEAU_SUPPORTED' if m=='vsoed' else 'INVALID_OR_INCOMPLETE' if m=='scorebed' else 'COMPUTE_LIMITED' if m in ['fixed','dad','rl_sboed'] else 'VALID_FIXED_COMPUTE'),evaluation_seed=cfg['seeds']['confirmation'] if s else '',evaluation_trajectories=2048 if s else '',evaluation_contrasts=1000000 if s else '',eig_mean=s.get('mean',''),eig_mcse=s.get('MC_SE',''),eig_ci_low=s.get('CI95_low',''),eig_ci_high=s.get('CI95_high',''),checkpoint_sha256=r.get('checkpoint',{}).get('sha256',''),source_hash=r.get('source_hash',sha(C/'manifests/source_sha256.json'))))
 with (C/'baseline_improvement_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (C/'reports/10_COMMON_EIG_EVALUATION.md').write_text('# Fresh common confirmation\n\n'+json.dumps(dict(results=result,paired=paired),indent=2));(C/'reports/12_TRAINING_SUFFICIENCY_COMPARISON.md').write_text('# Training-sufficiency comparison\n\n'+json.dumps(rows,indent=2)+'\nNo full convergence claim follows from time caps. Missing ScoreBED explicit; unchanged references re-evaluated on the same new bank.\n');(C/'reports/15_FINAL_SCIENTIFIC_ASSESSMENT.md').write_text('# Final assessment\n\n'+json.dumps(paired,indent=2)+'\nOne seed; MCSE excludes training-seed variance and residual finite-contrast bias. Frozen interventions/test results never used for optimizer selection.\n')

def acquire():
 fs=[]
 try:
  for p in [R.parent/'objective_power_grid/.gpu0.lock',Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock']:
   f=p.open('a+');fs.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if busy:raise BlockingIOError('GPU busy:'+busy)
  return fs
 except Exception:
  for f in fs:f.close()
  raise

def queue():
 f=(C/'manifests/controller.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (C/'manifests/controller_started.json').exists():raise RuntimeError('Existing scheduler; inspect before reviewed resume')
 write(C/'manifests/controller_started.json',dict(PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 def state(s,**kw):write(C/'status.json',dict(state=s,PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
 # Complete ORIGINAL DAD and its queued final evaluation before acquiring the GPU.
 while True:
  old=json.loads((O/'status.json').read_text());pid=old.get('PID');alive=pid and Path('/proc/'+str(pid)).exists()
  if old['state']=='COMPLETE_WITH_SCOREBED_COMPUTE_LIMIT':break
  if not alive or 'FAILED' in old['state']:write(C/'audits/original_completion_warning.json',old);break
  state('WAITING_ORIGINAL_DAD_AND_FINAL_EVALUATION',original=old);time.sleep(30)
 original_report=O/'sir_t3_recovery_results.csv'
 if original_report.exists():
  import shutil;shutil.copy2(original_report,C/'audits/original_fourhour_results.csv');(C/'reports/11_FOUR_HOUR_COMPARISON.md').write_text('# Immutable original four-hour comparison\n\n'+original_report.read_text())
 for phase in ['fixed','dad','rl_sboed','scorebed','final']:
  if phase=='scorebed':write(C/'manifests/scorebed_blocked.json',json.loads((C/'audits/scorebed_cost.json').read_text()));continue
  while True:
   try:lease=acquire();break
   except BlockingIOError as e:state('WAITING_GPU',next_phase=phase,reason=str(e));time.sleep(30)
  freeze_check();fds=[x.fileno() for x in lease];env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None);env['SIR_IMPROVE_LOCK_FDS']=','.join(map(str,fds));cmd=['bash','-c','source ./run.sh; exec python3 -m src.research.sir_t3_improvement "$1"','sir-improvement',phase]
  with (C/'logs'/f'{phase}.log').open('x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(fds),start_new_session=True)
  for x in lease:x.close()
  state('RUNNING',phase=phase,child_PID=child.pid)
  while child.poll() is None:time.sleep(30)
  write(C/'manifests'/f'{phase}_exit.json',dict(returncode=child.returncode))
  if child.returncode:state('STAGE_FAILED_CONTINUING_ELIGIBLE_METHODS',phase=phase,returncode=child.returncode)
  ledger=json.loads((C/'manifests/aggregate_budget.json').read_text())
  if ledger['active'] is not None:state('BLOCKED_INTERRUPTED_BUDGET_REVIEW',phase=phase);return
  assert sum(ledger['spent'].values())<=43200
  (C/'reports/13_COMPUTE_ACCOUNTING.md').write_text('# Additional charged training ledger\n\n'+json.dumps(ledger,indent=2))
 state('COMPLETE_WITH_RECORDED_LIMITATIONS');freeze_check()
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='queue':queue()
 elif phase=='fixed':fixed()
 elif phase=='dad':dad()
 elif phase=='rl_sboed':rl()
 elif phase=='final':final_evaluation()
 else:raise ValueError(phase)
