"""Unchanged native training; canonical EIG evaluation; measured resources."""
import copy,gc,json,os,random,resource,signal as signals,subprocess,threading,time
import numpy as np
import torch
from runtime import *
from model import StudyMoE
PROTOCOL=json.loads((HERE/'protocol.json').read_text());STOP=False
class BudgetStop(Exception):pass

def stop_handler(*unused):
 global STOP;STOP=True
signals.signal(signals.SIGTERM,stop_handler)
def guard(deadline):
 if STOP or time.monotonic()>deadline:raise BudgetStop('Four-hour bounded run stopping;partial evidence retained')
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
class Resources:
 def __init__(self,out):
  self.out=out;self.end=threading.Event();self.rows=[];self.start=time.monotonic();self.cpu0=resource.getrusage(resource.RUSAGE_SELF)
 def __enter__(self):
  hardware=dict(cpu_model=next((line.split(':',1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),logical_cpus=os.cpu_count())
  try:hardware['gpu_model_driver_total_memory']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
  except Exception as e:hardware['error']=str(e)
  put(self.out/'HARDWARE.json',hardware)
  self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def loop(self):
  with (self.out/'RESOURCE_SAMPLES.jsonl').open('x') as f:
   while not self.end.is_set():
    row=dict(seconds=time.monotonic()-self.start,pid=os.getpid());cpu=resource.getrusage(resource.RUSAGE_SELF)
    row.update(cpu_user_seconds=cpu.ru_utime-self.cpu0.ru_utime,cpu_system_seconds=cpu.ru_stime-self.cpu0.ru_stime,process_peak_rss_kb=cpu.ru_maxrss)
    try:
     value=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0]
     for k,v in zip(['gpu_utilization_percent','gpu_memory_mib','gpu_power_watts'],value.split(',')):
      try:row[k]=float(v.strip())
      except ValueError:row[k]=None
    except Exception as e:row['sampling_error']=str(e)
    self.rows.append(row);f.write(json.dumps(row)+'\n');f.flush();self.end.wait(5)
 def __exit__(self,*unused):
  self.end.set();self.thread.join(5);wall=time.monotonic()-self.start;cpu=resource.getrusage(resource.RUSAGE_SELF)
  util=[r['gpu_utilization_percent'] for r in self.rows if r.get('gpu_utilization_percent') is not None];mem=[r['gpu_memory_mib'] for r in self.rows if r.get('gpu_memory_mib') is not None]
  active=sum((b['seconds']-a['seconds'])*float(a.get('gpu_utilization_percent') or 0)/100 for a,b in zip(self.rows,self.rows[1:]))
  put(self.out/'RESOURCES.json',dict(wall_seconds=wall,cpu_user_seconds=cpu.ru_utime-self.cpu0.ru_utime,cpu_system_seconds=cpu.ru_stime-self.cpu0.ru_stime,
   process_peak_rss_kb=cpu.ru_maxrss,gpu_exclusive_wall_seconds=wall,sampled_gpu_busy_seconds_estimate=active,
   gpu_utilization_mean_percent=float(np.mean(util)) if util else None,gpu_peak_sampled_memory_mib=max(mem) if mem else None,
   torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   sampling='5-second nvidia-smi/process samples;busy estimate is not exact kernel time;GPU wall includes CPU/scorer/serialization gaps'))

def optimizer(a):
 a.actor_optimizer=torch.optim.Adam(list(a.actor_net.parameters()),lr=a.actor_optimizer.param_groups[0]['lr'])
 a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999)
def opts(a):return [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(o,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for o in oo]
def frozen_digest(model):
 d={k:p for k,p in model.named_parameters() if not p.requires_grad};d.update({'buffer:'+k:v for k,v in model.named_buffers()});return n.digest(d)
class Table:
 def __init__(self):self.data={}
 def record(self,k,v):self.data[k]=v
 def clear(self):self.data.clear()
class Logger:
 def __init__(self,a,out,name,base,steps,start,deadline):
  self.a=a;self.out=out;self.name=name;self.base=base;self.steps=steps;self.start=start;self.deadline=deadline;self.logger=self;self.tabular=Table();self.counts=dict(actor=0,critic=0,posterior=0);self.gradients={}
 def push_prefix(self,*unused):pass
 def pop_prefix(self,*unused):pass
 def log(self,*unused):pass
 def dump_all(self):
  done=int(self.a.update)-self.base;assert self.counts==dict(actor=done,critic=5*done,posterior=150*done)
  if done==1 or done%25==0 or done==self.steps:
   row=dict(status='TRAINING',phase=self.name,phase_updates=done,phase_target=self.steps,total_updates=int(self.a.update)-16001,counts=self.counts.copy(),seconds=time.monotonic()-self.start,
    native_iteration=int(self.a.update),actor_lr=self.a.actor_optimizer.param_groups[0]['lr'],gradient_group_norms=self.gradients)
   # Native library stores metrics in its internal table; no substitute reward is added here.
   with (self.out/'TRAINING_METRICS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   put(self.out/'STATUS.json',row)
  guard(self.deadline)
def train_phase(a,out,name,steps,start,deadline):
 a.actor_net.configure(name);base=int(a.update);logger=Logger(a,out,name,base,steps,start,deadline);a.dowel=logger;a.post_approx.dowel=logger;a.save_path=None
 for op,key in opts(a):
  op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
  def count(*unused,key=key):logger.counts[key]+=1
  op.register_step_post_hook(count)
 def gradient_check(*unused):
  groups={}
  for name,p in a.actor_net.named_parameters():
   if p.requires_grad:
    assert p.grad is not None and torch.isfinite(p.grad).all(),name
    group='.'.join(name.split('.')[:2]);groups[group]=groups.get(group,0.)+float(p.grad.square().sum())
   else:assert p.grad is None,name
  logger.gradients={k:v**.5 for k,v in groups.items()}
 a.actor_optimizer.register_step_pre_hook(gradient_check);before=frozen_digest(a.actor_net);status='COMPLETE';tick=time.monotonic()
 profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
 assert not a.use_PCE and a.discount==1 and profile['critic_updates']==5
 try:n.original.train_native(a,logger,profile,target=base+steps)
 except BudgetStop:status='INCOMPLETE'
 assert frozen_digest(a.actor_net)==before
 for op,_ in opts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 torch.save(dict(actor_state=a.actor_net.state_dict(),actor_target_state=a.actor_target_net.state_dict(),native_iteration=int(a.update)),out/(name+'_actor.pt'))
 receipt=dict(status=status,updates=int(a.update)-base,counts=logger.counts,seconds=time.monotonic()-tick,trainable_parameters=sum(p.numel() for p in a.actor_net.parameters() if p.requires_grad),
  new_episodes=1000*(int(a.update)-base),simulator_observations=30000*(int(a.update)-base),frozen_state_preserved=True,objective='Unmodified native variational actor-critic EIG;use_PCE=False')
 put(out/(name+'_TRAINING.json'),receipt)
 if status!='COMPLETE':raise BudgetStop('Incomplete native phase')
 return receipt

def build(seed,arm):
 state=parent();a=state['algorithm'];model=StudyMoE(a.actor_net,202611150000+seed,arm).cuda()
 a.actor_net=model;a.actor_target_net=copy.deepcopy(model);a.actor_target_net.configure('frozen')
 optimizer(a);n.original.restore(state['rng']);seed_all(seed);return state,a
@torch.no_grad()
def diagnostic_histories(model,seed,cases,deadline):
 rng=np.random.default_rng(seed);truth=rng.standard_normal((cases,4));noise=rng.standard_normal((cases,30));X=[];G=[];S=[];lib=n.load_library()
 for i in range(0,cases,32):
  th=torch.tensor(truth[i:i+32],device='cuda');eps=torch.tensor(noise[i:i+32],device='cuda');ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,device='cuda',dtype=torch.float64);xx=[];gg=[];ss=[]
  for t in range(30):
   guard(deadline);x=lib.packed(ds,ys,t);action,info=model(x,collect=True);ds[:,t]=action;ys[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*eps[:,t]
   xx.append(x.cpu().numpy());gg.append(info['gates'].cpu().numpy());ss.append(info['scale'].cpu().numpy())
  X.append(np.stack(xx,1));G.append(np.stack(gg,1));S.append(np.stack(ss,1))
 return dict(X=np.concatenate(X),gates=np.concatenate(G),scale=np.concatenate(S),truth=truth,noise=noise)

def summaries(values):
 from scipy.stats import t
 v=np.asarray(values,dtype=float);critical=float(t.ppf(.975,len(v)-1));se=float(v.std(ddof=1)/np.sqrt(len(v)));return dict(n=len(v),mean=float(v.mean()),MCSE=se,CI95=[float(v.mean()-critical*se),float(v.mean()+critical*se)])
@torch.no_grad()
def evaluate(model,out,label,deadline,collect=True):
 bank,u,scorer=canonical_load();lib=n.load_library();neg=torch.tensor(bank['contrasts'],device='cuda');values=[];curves=[];gates=[];firsts=[];innovations=[];responses=[];prefix_uncertainty=[]
 latency=0.;scoring=0.;observations=0;bench=type('Benchmark',(),{'sigma':.5,'calls':0})();start=time.monotonic();cpu0=resource.getrusage(resource.RUSAGE_SELF)
 for i in range(0,2000,16):
  guard(deadline);th=torch.tensor(bank['truth'][i:i+16],device='cuda');noise=torch.tensor(bank['noise'][i:i+16],device='cuda')
  ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,device='cuda',dtype=torch.float64);gg=[];ff=[];rr=[];ii=[]
  for t in range(30):
   x=lib.packed(ds,ys,t);torch.cuda.synchronize();tick=time.monotonic()
   if collect:action,info=model(x,collect=True)
   else:action=model(x)
   torch.cuda.synchronize();latency+=time.monotonic()-tick
   assert torch.isfinite(action).all() and (action.abs()<=4.000001).all()
   ds[:,t]=action;ys[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*noise[:,t];observations+=len(th)
   if collect and t==9:prefix_uncertainty.extend(info['scale'].square().sum(-1).cpu().numpy())
   if collect:
    gg.append(info['gates'].cpu().numpy());ff.append(info['first'].square().mean(-1).sqrt().cpu().numpy());rr.append(info['responses'].square().mean(-1).sqrt().cpu().numpy());ii.append(info['innovation'].square().mean(-1).sqrt().cpu().numpy())
  torch.cuda.synchronize();tick=time.monotonic()
  score,_=scorer(bench,dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th),neg,16384,'cuda');torch.cuda.synchronize();scoring+=time.monotonic()-tick
  values.extend(score[:,-1].numpy());curves.append(score.numpy())
  if collect:gates.append(np.stack(gg,1));firsts.append(np.stack(ff,1));responses.append(np.stack(rr,1));innovations.append(np.stack(ii,1))
  put(out/'STATUS.json',dict(status='EVALUATING',method=label,completed=min(i+16,2000),total=2000))
 values=np.asarray(values);np.save(out/(label+'_scores.npy'),values);np.save(out/(label+'_curves.npy'),np.concatenate(curves))
 result=dict(**summaries(values),seconds=time.monotonic()-start,online_actor_batch16_total_seconds=latency,online_amortized_seconds_per_design=latency/60000,
  online_amortized_seconds_per_episode=latency/2000,scorer_seconds=scoring,simulator_observations=observations,negative_likelihood_calls=int(bench.calls),
  latency_scope='CUDA-synchronized model forward incl shared belief/model queries;batch16 amortized timings,not single-case latency;scorer excluded.',protocol=LOCK)
 if collect:
  np.save(out/(label+'_prefix_uncertainty.npy'),np.asarray(prefix_uncertainty))
  g=np.concatenate(gates);np.savez_compressed(out/(label+'_routing.npz'),gates=g,first_norm=np.concatenate(firsts),response_norm=np.concatenate(responses),innovation_norm=np.concatenate(innovations))
  result.update(stage_mean_weights=g.mean(0).tolist(),stage_std_weights=g.std(0,ddof=1).tolist())
 put(out/(label+'_EVALUATION.json'),result);return values,result

@torch.no_grad()
def configure_intervention(model,label,table=None):
 model.correction_enabled=model.arm in ['learned','router_only'];model.encoders_enabled=model.arm in ['learned','uniform','encoders_only'];model.gate_table=None;model.observation_swap=False;model.remove_feature=None
 if label=='uniform':model.gate_table=torch.full((30,4),.25,device='cuda')
 if label in ['no_correction','no_correction_no_encoders']:model.correction_enabled=False
 if label in ['no_encoders','no_correction_no_encoders']:model.encoders_enabled=False
 if label=='stage_only':model.gate_table=torch.tensor(table,device='cuda',dtype=torch.float32)
 if label=='observation_swapped_gates':model.observation_swap=True
 if label.startswith('remove_feature_'):model.remove_feature=int(label[-1])-1

def run_training(seed,arm,out):
 start=time.monotonic();out.mkdir();source=snapshot_sources();put(out/'SOURCE_MANIFEST.json',source);put(out/'CONFIG.json',dict(protocol_sha256=sha(HERE/'protocol.json'),seed=seed,arm=arm,model_seed=202611150000+seed,config=PROTOCOL))
 setup()
 with lease(),Resources(out):
  try:
   verify_references();state,a=build(seed,arm);COST.clear();torch.cuda.reset_peak_memory_stats();put(out/'INITIALIZATION.json',dict(parent_update=int(a.update),parent_actor_sha256=n.ACTOR_SHA,parent_runtime_sha256=n.RUNTIME_SHA,actor_digest=n.digest(a.actor_net.state_dict()),initial_actor_lr=a.actor_optimizer.param_groups[0]['lr'],new_branch_seed=202611150000+seed,arm=arm))
   trainstart=time.monotonic();phases=[]
   for phase,steps in [('warmup',200),('joint',1800)]:phases.append(train_phase(a,out,phase,steps,start,start+12600))
   training_seconds=time.monotonic()-trainstart
   # Save complete native algorithm/optimizer/replay/RNG state, without changing historical parents.
   with (out/'runtime.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng(),seed=seed,arm=arm),f,protocol=5)
   actor=a.actor_net;a.actor_net.configure('frozen');actor.eval();model_calls=dict(COST);del state,a;gc.collect();torch.cuda.empty_cache()
   diagnostic=None;table=None
   if arm=='learned':
    diagnostic=diagnostic_histories(actor,PROTOCOL['diagnosis_bank']['seed']+seed,1024,start+14100);np.savez_compressed(out/'diagnostic_histories.npz',**diagnostic);table=diagnostic['gates'].mean(0)
   labels=PROTOCOL['inference_ablations'] if arm=='learned' else [arm];scores={};evaluations={}
   for label in labels:
    configure_intervention(actor,label,table);scores[label],evaluations[label]=evaluate(actor,out,label,start+14100,collect=label in ['learned','uniform','v2_control','router_only','encoders_only'])
   contrasts={k:paired(scores['learned'],v) for k,v in scores.items() if k!='learned'} if arm=='learned' else {}
   assert snapshot_sources()==source and sha(n.BASE/'arms/candidate/actor.pt')==n.ACTOR_SHA and sha(n.BASE/'arms/candidate/runtime.pkl')==n.RUNTIME_SHA
   result=dict(status='COMPLETE',seed=seed,arm=arm,updates=2000,phases=phases,training_seconds=training_seconds,total_seconds=time.monotonic()-start,training_model_calls=model_calls,
    new_training_episodes=2000000,new_training_observations=60000000,diagnostic_observations=30720 if arm=='learned' else 0,evaluations=evaluations,inference_comparisons=contrasts,
    actor_sha256=sha(out/'joint_actor.pt'),runtime_sha256=sha(out/'runtime.pkl'),historical_artifacts_preserved=True,scientific_status='DEVELOPMENT_UNVALIDATED;one continuation seed')
   put(out/'RESULTS.json',result);relocation_receipt(out/'RELOCATION.json');put(out/'STATUS.json',dict(status='COMPLETE',seed=seed,arm=arm,seconds=time.monotonic()-start))
  except BudgetStop as e:put(out/'STATUS.json',dict(status='INCOMPLETE',error=str(e),seconds=time.monotonic()-start));raise
  except Exception as e:put(out/'STATUS.json',dict(status='FAILED',error=repr(e),seconds=time.monotonic()-start));raise
