"""Revised isolated SIR improvement: max6additional optimization/training hours, numerical profiling separate, no preemption."""
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
R=Path(__file__).resolve().parents[2];C=R/'scratch_runs/sir_t3_dad_fixed_correction';cfg=json.loads((C/'configs/protocol.json').read_text());P=Path(cfg['parent']);O=Path(cfg['original_extension'])
def freeze_check():
 for rel,h in json.loads((C/'manifests/source_sha256.json').read_text()).items():assert sha(R/rel)==h,'Source changed: '+rel
 for path,h in json.loads((C/'manifests/original_preservation_sha256.json').read_text()).items():assert sha(path)==h,'Original artifact changed: '+path

def leases():
 ids=[int(v) for v in os.environ.get('SIR_IMPROVE_LOCK_FDS','').split(',') if v];assert len(ids)==3,'Worker requires exclusive GPU lease';return [os.fdopen(os.dup(v),'a+') for v in ids]
class Budget:
 def __init__(self,method):
  self.method=method;self.p=C/'manifests'/('aggregate_budget.json' if method in ['fixed','dad'] else 'numerical_budget.json');x=json.loads(self.p.read_text());assert x['active'] is None,'Review interrupted stage before recovery';self.prior=x['spent'][method];self.start=ENTRY;x['active']=dict(method=method,start=ENTRY,prior=self.prior,PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));write(self.p,x);self.bound=2.
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


def fixed_audit():
 lease=leases();freeze_check();budget=Budget('fixed');torch.set_num_threads(4)
 try:
  with torch.device('cuda:0'):
   b=T3SIR(cfg['benchmark']);old=json.loads((C/'audits/fixed_cpu_correctness.json').read_text());schedules=[old['old_fixed_times'],[25.,50.,75.],[5.,10.,20.],[15.,30.,50.],[70.,85.,99.],[.1001,.2002,99.999]];g=torch.Generator(device='cpu').manual_seed(cfg['seeds']['candidate_generation']);u=torch.rand(8,3,generator=g,dtype=torch.float64,device='cpu').sort(-1).values;schedules+=[(.1*torch.arange(1,4,dtype=torch.float64,device='cpu')+99.7*x).tolist() for x in u];scores=[]
   for i,t in enumerate(schedules):
    p=schedule_policy(b,t).cuda()
    with torch.no_grad():a=fixed_shared_spce(b,p,128,65536,cfg['seeds']['schedule_diagnostics'])
    scores.append(a.cpu().numpy())
   original=scores[0];rows=[dict(times=t,EIG=stats(v),paired_minus_original=stats(v-original)) for t,v in zip(schedules,scores)];better=[x for x in rows[1:] if x['paired_minus_original']['CI95_low']>0];classification='OPTIMIZATION_FAILURE' if len(better)>=2 else 'UNRESOLVED';result=dict(classification=classification,coding_bug_identified=False,mathematical_and_gradient_tests='PASS_CPU',development_seed=cfg['seeds']['schedule_diagnostics'],generation_seed=cfg['seeds']['candidate_generation'],N=128,L=65536,rows=rows,meaning='Joint sPCE across fixed schedules with common prior/noise/contrast draws; development evidence only. Historical final scores never used for selection.');write(C/'audits/fixed_schedule_diagnostics.json',result);(C/'reports/FIXED_ROOT_CAUSE.md').write_text('# Fixed root cause\n\n'+json.dumps(result,indent=2));(C/'reports/FIXED_SCHEDULE_DIAGNOSTICS.md').write_text('# Development-only schedule diagnostics\n\n'+json.dumps(rows,indent=2));budget.finish('FIXED_AUDIT_COMPLETE')
 except Exception as e:budget.finish('FAILED');write(C/'manifests/fixed_audit_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise


def fixed():
 lease=leases();freeze_check();budget=Budget('fixed');torch.set_num_threads(4)
 try:
  with torch.device('cuda:0'):
   b=T3SIR(cfg['benchmark']);g=torch.Generator(device='cpu').manual_seed(cfg['seeds']['candidate_generation']);U=torch.rand(64,3,generator=g,dtype=torch.float64,device='cpu').sort(-1).values;times=.1*torch.arange(1,4,dtype=torch.float64,device='cpu')+(100-.3)*U
   # Half the starts cover early/medium schedules; all lie in the same feasible space.
   times[32:]=.1*torch.arange(1,4,dtype=torch.float64,device='cpu')+(40-.3)*U[32:];times[1]=torch.tensor([25.,50.,75.],device='cpu');times[2]=torch.tensor([5.,10.,20.],device='cpu');times[3]=torch.tensor([15.,30.,50.],device='cpu');times[4]=torch.tensor([70.,85.,99.],device='cpu');times[0]=torch.tensor(json.loads((C/'audits/fixed_cpu_correctness.json').read_text())['old_fixed_times'],device='cpu');screen=[]
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
   best=int(np.argmax(values));cp=checkpoint('fixed_corrected',models[best],opts[best],sched[best],u,budget,extra=dict(selected_start=best,selection_development_means=values,selection_seed=cfg['fixed']['selection_seed'],times=physical_times(b,models[best])));total=budget.finish('MULTISTART_PROTOCOL_COMPLETE');result=dict(method='fixed',variant='corrected_multistart_joint_spce',training_seed=101,training_objective='joint full-horizon sPCE',batch_size=512,training_contrasts=16384,optimizer_updates=u*8,selected_optimizer_updates=u,fresh_training_trajectories=u*8*512+64*128+8*512,charged_training_seconds=total,additional_training_seconds=total,native_schedule_complete=False,convergence_status='COMPUTE_LIMITED',checkpoint=cp,times=physical_times(b,models[best]),selection_development_means=values,source_hash=sha(C/'manifests/source_sha256.json'));write(C/'manifests/fixed_complete.json',result);(C/'reports/FIXED_MULTISTART_OPTIMIZATION.md').write_text('# Corrected Fixed result\n\n'+json.dumps(result,indent=2))
 except Exception as e:budget.finish('FAILED');write(C/'manifests/fixed_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

def dad_profile():
 lease=leases();freeze_check();budget=Budget('dad_profile');torch.set_num_threads(4)
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
   rows=[];policy_grads=[];infos=[];profile_summary=[];likelihoods=[];timegrads=[];observations=[]
   for name,cls in [('reference',T3SIR),('count_sorted',CountSortedSIR)]:
    for rep in range(2):
     p=copy.deepcopy(saved['policy']).cuda().train();b=cls(cfg['benchmark']);p.zero_grad();optimizer=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));cpu_start=time.process_time();torch.cuda.reset_peak_memory_stats();prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if name=='reference' and rep==0 else __import__('contextlib').nullcontext();prof.__enter__();oldforward=p.forward;oldprop=b.propagate;oldll=b.log_likelihood
     def traced_forward(*args,**kw):
      with torch.profiler.record_function('DAD_POLICY_FORWARD'):return oldforward(*args,**kw)
     def traced_prop(*args,**kw):
      with torch.profiler.record_function('SIR_ODE_PROPAGATION'):return oldprop(*args,**kw)
     def traced_ll(*args,**kw):
      with torch.profiler.record_function('GAUSSIAN_LOG_LIKELIHOOD'):return oldll(*args,**kw)
     if name=='reference' and rep==0:p.forward=traced_forward;b.propagate=traced_prop;b.log_likelihood=traced_ll
     torch.cuda.synchronize();t=time.monotonic();r=rollout(b,p,3,2000,cfg['seeds']['numerics']+1,2000);torch.cuda.synchronize();forward=time.monotonic()-t;info=r['info'][:,-1];loss=-info.mean()
     for a in r['actions']:a.retain_grad()
     t=time.monotonic();loss.backward();torch.cuda.synchronize();back=time.monotonic()-t;prof.__exit__(None,None,None)
     if 'forward' in p.__dict__:del p.__dict__['forward']
     if name=='reference' and rep==0:
      for event in prof.key_averages():profile_summary.append(dict(name=event.key,calls=event.count,CPU_us=event.cpu_time_total,device_us=getattr(event,'device_time_total',0)))
      prof.export_chrome_trace(str(C/'audits/dad_reference_trace.json'))
     pg=torch.cat([x.grad.flatten() for x in p.parameters()]).detach().cpu();optstart=time.monotonic();optimizer.step();torch.cuda.synchronize();optseconds=time.monotonic()-optstart;savebegin=time.monotonic();torch.save(dict(policy=copy.deepcopy(p).cpu(),optimizer=optimizer.state_dict()),C/'audits'/f'{name}_{rep}_roundtrip.pt');saveseconds=time.monotonic()-savebegin;rows.append(dict(implementation=name,repeat=rep,forward_seconds=forward,backward_seconds=back,optimizer_seconds=optseconds,checkpoint_seconds=saveseconds,CPU_process_seconds=time.process_time()-cpu_start,peak_VRAM_bytes=torch.cuda.max_memory_allocated(),ODE_particle_stages=b.calls,effective_RK4_steps=b.ode_forward_steps,likelihood_evaluations=2000*2001*3,total_seconds=forward+back+optseconds,training_sPCE=float(info.mean().detach()),saturation_fraction=float((info>math.log(2001)-.1).double().mean())));policy_grads.append(pg);infos.append(info.detach().cpu());likelihoods.append(r['log_likelihood'].detach().cpu());timegrads.append(torch.stack([a.grad for a in r['actions']]).detach().cpu());observations.append(torch.stack(r['observations']).detach().cpu());del r,loss,info,p
   torch.testing.assert_close(infos[1],infos[3],rtol=1e-8,atol=1e-7);torch.testing.assert_close(likelihoods[1],likelihoods[3],rtol=1e-8,atol=1e-7);torch.testing.assert_close(timegrads[1],timegrads[3],rtol=1e-7,atol=1e-6);torch.testing.assert_close(observations[1],observations[3],rtol=1e-8,atol=1e-7);torch.testing.assert_close(policy_grads[1],policy_grads[3],rtol=1e-7,atol=1e-6);speedup=rows[1]['total_seconds']/rows[3]['total_seconds'];validation=dict(status='PASS',physical_predictions=True,loglikelihood_and_terminal_sPCE=True,time_parameter_initial_state_gradients=True,policy_gradients=True,high_rate_boundary_cases=True,records=rows,profiler=profile_summary,speedup=speedup,sort_only_no_physics_change=True);write(C/'audits/dad_gpu_validation.json',validation);(C/'reports/DAD_ACCELERATION.md').write_text('# Measured acceleration\n\n'+json.dumps(dict(speedup=speedup,required_minimum=cfg['dad']['minimum_speedup'],reference_seconds=rows[1]['total_seconds'],candidate_seconds=rows[3]['total_seconds']),indent=2));(C/'reports/DAD_RUNTIME_PROFILE.md').write_text('# Native DAD profile\n\n'+json.dumps(validation,indent=2));(C/'reports/DAD_NUMERICAL_EQUIVALENCE.md').write_text('# Numerical/gradient validation\n\n'+json.dumps(validation,indent=2))
   if speedup<cfg['dad']['minimum_speedup']:total=budget.finish('BLOCKED_NO_DEMONSTRATED_ACCELERATION');write(C/'manifests/dad_blocked.json',dict(reason='Measured speedup below predeclared substantial-acceleration gate; no essentially unchanged undertrained run launched',speedup=speedup,charged_seconds=total));return
   resolution=[];resolution_gradients=[]
   for rep in range(4):
    p=copy.deepcopy(saved['policy']).cuda();b=CountSortedSIR(cfg['benchmark']);p.zero_grad();r=rollout(b,p,3,128,cfg['seeds']['numerics']+100+rep,2000);info=r['info'][:,-1];(-info.mean()).backward();resolution_gradients.append(torch.cat([q.grad.flatten() for q in p.parameters()]).detach().cpu());resolution.append(dict(B=128,L=2000,mean_sPCE=float(info.mean().detach()),distance_to_ceiling=math.log(2001)-float(info.mean().detach()),fraction_within_point1_of_ceiling=float((info>math.log(2001)-.1).double().mean()),gradient_norm=float(torch.cat([q.grad.flatten() for q in p.parameters()]).norm())));del r,info,p
   write(C/'audits/dad_training_resolution.json',dict(records=resolution,mean_coordinate_gradient_variance=float(torch.stack(resolution_gradients).var(0,unbiased=True).mean()),scope='Independent diagnostic minibatches, not nativeB2000 gradient variance or final EIG',higher_contrast_training_launched=False));(C/'reports/DAD_TRAINING_RESOLUTION.md').write_text('# Training contrast diagnostics\n\n'+json.dumps(resolution,indent=2)+'\nNativeL2000 retained. Saturation is diagnostic, not automatic evidence of its being the bottleneck. Any higher-L study must be separately named.\n')
   budget.finish('NUMERICAL_VALIDATION_COMPLETE');write(C/'manifests/dad_profile_ready.json',dict(status='PASS',speedup=speedup,unit_seconds=rows[3]['total_seconds'],reference_checkpoint=cpold))
 except Exception as e:budget.finish('FAILED');write(C/'manifests/dad_profile_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

def dad():
 lease=leases();freeze_check()
 ready=C/'manifests/dad_profile_ready.json'
 if not ready.exists():write(C/'manifests/dad_blocked.json',dict(reason='No passing validated acceleration; no corrected training launched'));return
 budget=Budget('dad');torch.set_num_threads(4)
 try:
  with torch.device('cuda:0'):
   from src.research.sir_dad import SIRSumDAD
   torch.manual_seed(101);p=SIRSumDAD(3);opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,.98);b=CountSortedSIR(cfg['benchmark']);u=0;nextcp=budget.spent()+300;unit=json.loads(ready.read_text())['unit_seconds'];history=[];nextdev=3600
   while u<50000 and budget.can(unit,reserve=120):
    u+=1;t=time.monotonic();opt.zero_grad();r=rollout(b,p,3,2000,101+u*104729,2000);info=r['info'][:,-1];loss=-info.mean();assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(p.parameters(),float('inf'),error_if_nonfinite=True);opt.step()
    if (u-1)%1000==0:scheduler.step()
    torch.cuda.synchronize();unit=max(unit,time.monotonic()-t)
    if u==1 or u%25==0:
     row=dict(update=u,charged_seconds=budget.spent(),training_sPCE=float(info.mean().detach()),distance_to_log2001=math.log(2001)-float(info.mean().detach()),saturation_fraction=float((info>math.log(2001)-.1).double().mean()),gradient_norm=float(grad),lr=opt.param_groups[0]['lr']);history.append(row);write(C/'logs/dad_learning.json',history);budget.status('CORRECTED_NATIVE_DAD_TRAINING',optimizer_updates=u,fresh_trajectories=u*2000)
    if u==1 or budget.spent()>=nextcp:checkpoint('dad_accelerated',p,opt,scheduler,u,budget,extra=dict(initialization='fresh seed101',native_clipping='none',batch=2000,contrasts=2000));nextcp=budget.spent()+300
    if budget.spent()>=nextdev and budget.can(unit,reserve=180):development('dad',p,u,budget);nextdev+=3600
   cp=checkpoint('dad_accelerated',p,opt,scheduler,u,budget,extra=dict(initialization='fresh seed101'));assert cp['eligible'];total=budget.finish('COMPLETE');result=dict(method='dad',variant='count_sorted_exact_native_seed101_from_scratch',training_seed=101,training_objective='full-horizon native sPCE',batch_size=2000,training_contrasts=2000,optimizer_updates=u,fresh_training_trajectories=u*2000,charged_training_seconds=total,additional_training_seconds=total,native_schedule_complete=u>=50000,convergence_status='NATIVE_SCHEDULE_COMPLETE' if u>=50000 else 'COMPUTE_LIMITED',checkpoint=cp,source_hash=sha(C/'manifests/source_sha256.json'));write(C/'manifests/dad_complete.json',result);(C/'reports/DAD_CORRECTED_TRAINING.md').write_text('# Fresh corrected native DAD\n\n'+json.dumps(result,indent=2))
 except Exception as e:budget.finish('FAILED');write(C/'manifests/dad_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

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
 for m in ['random','fixed','dad','rl_sboed','vsoed']:
  r=records.get(m,{});s=result.get(m,{});variant=r.get('variant','original_fourhour_reference' if m!='scorebed' else 'SCOREBED_NATIVE_BUDGET_INFEASIBLE');rows.append(dict(method=m,variant=variant,training_seed=101,objective=r.get('training_objective','no training' if m=='random' else 'native original'),batch_size=r.get('batch_size',''),training_contrasts=r.get('training_contrasts',r.get('contrasts','')),optimizer_updates=r.get('optimizer_updates',0),training_trajectories=r.get('fresh_training_trajectories',r.get('generated_training_trajectories',0)),charged_training_seconds=r.get('charged_training_seconds',r.get('charged_training',{}).get('charged_seconds',0)),additional_training_seconds=r.get('additional_training_seconds',0),native_schedule_complete=r.get('native_schedule_complete',m=='random'),convergence_status=r.get('convergence_status','PLATEAU_SUPPORTED' if m=='vsoed' else 'INVALID_OR_INCOMPLETE' if m=='scorebed' else 'COMPUTE_LIMITED' if m in ['fixed','dad','rl_sboed'] else 'VALID_FIXED_COMPUTE'),evaluation_seed=cfg['seeds']['confirmation'] if s else '',evaluation_trajectories=2048 if s else '',evaluation_contrasts=1000000 if s else '',eig_mean=s.get('mean',''),eig_mcse=s.get('MC_SE',''),eig_ci_low=s.get('CI95_low',''),eig_ci_high=s.get('CI95_high',''),checkpoint_sha256=r.get('checkpoint',{}).get('sha256',''),source_hash=r.get('source_hash',r.get('source_amendment_sha256',r.get('source_manifest_sha256',sha(C/'manifests/source_sha256.json')))),t1=r.get('times',['','',''])[0],t2=r.get('times',['','',''])[1],t3=r.get('times',['','',''])[2]))
 for tag in ['fixed_original_4h','dad_original_4h']:
  if tag in result:
   r=records[tag];ss=result[tag];rows.append(dict(method=tag.split('_original')[0],variant='original_fourhour_on_fresh_confirmation',training_seed=101,objective='original joint full-horizon sPCE',batch_size=2000,training_contrasts=r.get('contrasts',2000),optimizer_updates=r['optimizer_updates'],training_trajectories=r.get('generated_training_trajectories',0),charged_training_seconds=r['charged_training']['charged_seconds'],additional_training_seconds=0,native_schedule_complete=r.get('native_schedule_complete',False),convergence_status='COMPUTE_LIMITED',evaluation_seed=cfg['seeds']['confirmation'],evaluation_trajectories=2048,evaluation_contrasts=1000000,eig_mean=ss['mean'],eig_mcse=ss['MC_SE'],eig_ci_low=ss['CI95_low'],eig_ci_high=ss['CI95_high'],checkpoint_sha256=r['checkpoint']['sha256'],source_hash=r.get('source_amendment_sha256',r.get('source_manifest_sha256','')),t1='',t2='',t3=''))
 with (C/'dad_fixed_correction_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 
 for method in ['fixed','dad']:
  if method in result:(C/'reports'/('FIXED_FINAL_EIG.md' if method=='fixed' else 'DAD_FINAL_EIG.md')).write_text('# Fresh final EIG\n\n'+json.dumps(dict(result=result[method],paired={k:v for k,v in paired.items() if k.startswith(method+'-')}),indent=2))
 (C/'reports/COMMON_EIG_COMPARISON.md').write_text('# Fresh common confirmation\n\n'+json.dumps(dict(results=result,paired=paired),indent=2));(C/'reports/FINAL_SCIENTIFIC_DECISION.md').write_text('# Training-sufficiency comparison\n\n'+json.dumps(rows,indent=2)+'\nNo full convergence claim follows from time caps. RL/vsOED are unchanged read-only references re-evaluated on the same new bank; no training in this campaign.\n');(C/'reports/PAIRED_COMPARISONS.md').write_text('# Final assessment\n\n'+json.dumps(paired,indent=2)+'\nOne seed; MCSE excludes training-seed variance and residual finite-contrast bias. Frozen interventions/test results never used for optimizer selection.\n')

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
  import shutil;shutil.copy2(original_report,C/'audits/original_fourhour_results.csv');(C/'reports/ORIGINAL_FOUR_HOUR_COMPARISON.md').write_text('# Immutable original four-hour comparison\n\n'+original_report.read_text())
 for phase in ['fixed_audit','dad_profile','fixed','dad','final']:
  while True:
   try:lease=acquire();break
   except BlockingIOError as e:state('WAITING_GPU',next_phase=phase,reason=str(e));time.sleep(30)
  freeze_check();fds=[x.fileno() for x in lease];env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None);env['SIR_IMPROVE_LOCK_FDS']=','.join(map(str,fds));cmd=['bash','-c','source ./run.sh; exec python3 -m src.research.sir_t3_dad_fixed "$1"','sir-improvement',phase]
  with (C/'logs'/f'{phase}.log').open('x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(fds),start_new_session=True)
  for x in lease:x.close()
  state('RUNNING',phase=phase,child_PID=child.pid)
  while child.poll() is None:
   ledger=json.loads((C/'manifests/aggregate_budget.json').read_text());active=ledger['active']
   if active and active['method'] in ['fixed','dad'] and active['prior']+time.monotonic()-active['start']>=ledger['caps'][active['method']]-45:
    os.kill(child.pid,signal.SIGTERM);child.wait(timeout=30);ledger['spent'][active['method']]=active['prior']+time.monotonic()-active['start'];assert ledger['spent'][active['method']]<=ledger['caps'][active['method']];ledger['active']=None;ledger.setdefault('history',[]).append(dict(method=active['method'],status='PROCESS_SAFETY_CUTOFF',charged_seconds=ledger['spent'][active['method']]));write(C/'manifests/aggregate_budget.json',ledger);break
   time.sleep(10)
  write(C/'manifests'/f'{phase}_exit.json',dict(returncode=child.returncode))
  if child.returncode:state('STAGE_FAILED_CONTINUING_ELIGIBLE_METHODS',phase=phase,returncode=child.returncode)
  ledger=json.loads((C/'manifests/aggregate_budget.json').read_text())
  if ledger['active'] is not None:state('BLOCKED_INTERRUPTED_BUDGET_REVIEW',phase=phase);return
  assert sum(ledger['spent'].values())<=21600
  (C/'reports/COMPUTE_ACCOUNTING.md').write_text('# Additional charged training ledger\n\n'+json.dumps(dict(optimization_training=ledger,numerical=json.loads((C/'manifests/numerical_budget.json').read_text())),indent=2))
 state('COMPLETE_WITH_RECORDED_LIMITATIONS');freeze_check()
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='queue':queue()
 elif phase=='fixed':fixed()
 elif phase=='dad':dad()
 elif phase=='dad_profile':dad_profile()
 elif phase=='fixed_audit':fixed_audit()
 elif phase=='final':final_evaluation()
 else:raise ValueError(phase)
