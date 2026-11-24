import os,sys,pathlib,json,time,hashlib,copy,random,signal,subprocess,fcntl,traceback,csv,math
import numpy as np
import torch,cloudpickle
from torch import nn
HERE=pathlib.Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
sys.path.insert(0,str(HERE))
from moe_actor import DirectDesignHeads,inject,digest
from src.research.vendor.vsoed import PGvsOED,POST_APPROX
from src.research.vendor.vsoed.models import SOURCE
from src.research.vendor.vsoed.utils import set_random_seed
from src.research.upstream_vsoed import UpstreamActor
STOP=False
class SafeStop(Exception):pass
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def put(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False));q.replace(p)
def setup():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
def verify():
 m=read(OUT/'source_manifest.json');assert sha(OUT/'frozen_protocol.json')==m['protocol_sha256']
 for p,h in m['files'].items():assert sha(OUT/p)==h,p
 provenance=read(OUT/'reference/baselines/provenance.json')
 assert sha(HERE/'src/research/location_evaluation.py')==provenance['scorer_sha256']
 assert sha(HERE/'src/research/engine.py')==provenance['engine_sha256']
 for filename,h in provenance['files'].items():assert sha(OUT/'reference/baselines'/filename)==h
def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],numpy=np.random.get_state(),python=random.getstate())
def restore(r):
 torch.set_rng_state(r['torch'].cpu());np.random.set_state(r['numpy']);random.setstate(r['python'])
 if r['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def acquire():
 held=[]
 try:
  for p in [ROOT/'.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',ROOT.parent/'objective_power_grid/.gpu0.lock']:
   if p.parent.name=='objective_power_grid':
    if not p.exists():continue
    f=p.open('r')
   else:p.parent.mkdir(exist_ok=True,parents=True);f=p.open('a+')
   held.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if active:raise BlockingIOError(active)
  return held
 except:
  for f in held:f.close()
  raise
class Table:
 def __init__(self):self.data={}
 def record(self,k,v):self.data[k]=v
 def clear(self):self.data.clear()
class Logger:
 def __init__(self,smoke=False):
  self.tabular=Table();self.logger=self;self.algorithm=None;self.smoke=smoke;self.start=time.monotonic();self.credit=0.;self.counts=dict(actor=0,critic=0,posterior=0);self.saving=False;self.last_saved=-1
 def push_prefix(self,*a):pass
 def pop_prefix(self,*a):pass
 def log(self,*a):pass
 def dump_all(self):
  a=self.algorithm
  if a is None or not hasattr(a,'_runtime_buffer'):return
  u=int(a.update);elapsed=self.credit+time.monotonic()-self.start
  if self.smoke:return
  if u==1 or u%100==0:
   block=a.actor_net.net[4];usage=block.last_probs.reshape(-1,4).mean(0).cpu().tolist()
   record=dict(status='RUNNING',successful_iterations=u,attempted_iterations=u,actor_optimizer_updates=self.counts['actor'],critic_optimizer_updates=self.counts['critic'],posterior_optimizer_updates=self.counts['posterior'],fresh_primary_trajectories=20000+u*1000,prior_inherited_trajectories=0,wall_seconds=elapsed,learning_rate=a.actor_optimizer.param_groups[0]['lr'],exploration=float(a.design_noise_scale),head_weight_means=usage,gate_total_mass=float(block.last_probs.sum(-1).mean()),replay_size=len(a._runtime_buffer['params']),replay_batch=10000,reward=self.tabular.data.get('Reward/MeanReward'),gpu_peak_bytes=torch.cuda.max_memory_allocated(),pid=os.getpid())
   assert all(torch.isfinite(v).all() for v in a.actor_net.parameters()),'nonfinite actor parameters'
   put(OUT/'status.json',record)
   with open(OUT/'logs/metrics.jsonl','a') as f:f.write(json.dumps(record)+'\n')
   with open(OUT/'compute_accounting.csv','w') as f:
    w=csv.DictWriter(f,['successful_iterations','actor_optimizer_updates','critic_optimizer_updates','posterior_optimizer_updates','fresh_primary_trajectories','wall_seconds','replay_size','replay_batch']);w.writeheader();w.writerow({k:record[k] for k in w.fieldnames})
   print(json.dumps(record),flush=True)
  if u%1000==0 or u==10001 or STOP or (OUT/'STOP_REQUESTED').exists() or elapsed>=41400:
   save_runtime(self,OUT/'checkpoints'/f'update{u}_runtime.pkl')
  if STOP or (OUT/'STOP_REQUESTED').exists() or elapsed>=41400:raise SafeStop()
def build(profile,device='cpu',candidate=True,seed=101):
 with torch.device(device):
  set_random_seed(seed);model=SOURCE(multimodel=False,include_nuisance=False,include_goal=False);logger=Logger(smoke=True)
  post=POST_APPROX(n_stage=30,n_design=2,n_obs=1,n_pois=[4],model_weight=0,poi_weight=1,goal_weight=0,n_mixture=8,mu_bounds={'poi':[[[-6,6]]*4]},max_sigmas={'poi':[[1.]*4]},truncnorm_info={'poi':[None]},prior=model,n_incre=30,share_interm_net=False,use_NFs=False,activate=nn.ReLU,poi_post_lrs=[profile['posterior_lr']]*30,poi_post_lr_scheduler_gammas=[profile['lr_decay']]*30,log_every=100,dowel=logger)
  a=PGvsOED(30,4,2,1,model,model,[(-4.,4.)]*2,model.reward_fun,post.reward_fun,post_approx=post,actor_dimns=[256]*3,critic_dimns=[256]*3,activate=nn.ReLU)
  init=inject(a) if candidate else {};logger.algorithm=a
  actor_opt=torch.optim.Adam(a.actor_net.parameters(),lr=profile['actor_lr']);critic_opt=torch.optim.Adam(a.critic_net.parameters(),lr=profile['critic_lr'])
  assert len({id(p) for g in actor_opt.param_groups for p in g['params']})==len(list(a.actor_net.parameters()))
  assert {id(p) for g in actor_opt.param_groups for p in g['params']}=={id(p) for p in a.actor_net.parameters()}
  a.actor_optimizer=actor_opt;a.critic_optimizer=critic_opt
  a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(actor_opt,profile['lr_decay']);a.critic_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(critic_opt,profile['lr_decay'])
  return a,logger,init
def attach(a,logger):
 for o,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(o,'posterior') for opts in a.post_approx.poi_post_optimizers.values() for o in opts]:
  def count(*args,key=key,**kwargs):logger.counts[key]+=1
  o.register_step_post_hook(count)
def train_native(a,logger,p,target=None):
 with torch.device(next(a.actor_net.parameters()).device):
  a.train(n_update=p['updates'] if target is None else target,n_newtraj=p['new_trajectories'],n_batch=p['replay_batch'],n_buffer_init=p['initial_replay'],n_buffer_max=p['replay_capacity'],buffer_device=next(a.actor_net.parameters()).device,discount=p['discount'],n_critic_update=p['critic_updates'],n_post_approx_update=p['posterior_updates'],actor_optimizer=a.actor_optimizer,critic_optimizer=a.critic_optimizer,actor_lr_scheduler=a.actor_lr_scheduler,critic_lr_scheduler=a.critic_lr_scheduler,lr_target=p['target_rate'],design_noise_scale=p['exploration'],design_noise_decay=p['exploration_decay'],transition=p['transition'],on_policy=False,use_PCE=False,frozen=-1,log_every=p['log_every'],dowel=logger,save_path=None)
def save_runtime(logger,path):
 a=logger.algorithm;u=int(a.update);r=rng();logger.saving=True
 state=dict(method='direct_head_moe_vsoed_seed101',algorithm=a,logger=logger,rng=r,successful_iterations=u,elapsed_before_serialization=logger.credit+time.monotonic()-logger.start,source_sha256=sha(OUT/'source_manifest.json'),protocol_sha256=sha(OUT/'frozen_protocol.json'))
 tmp=path.with_suffix('.tmp')
 with open(tmp,'wb') as f:cloudpickle.dump(state,f,protocol=5)
 tmp.replace(path);logger.saving=False;logger.last_saved=u
 receipt=dict(path=str(path),sha256=sha(path),successful_iterations=u,charged_seconds=logger.credit+time.monotonic()-logger.start,full_runtime=True,includes='actor/target,critic/target,posterior,all optimizer/schedulers,replay,p,p_max,RNG,exploration,counters')
 put(path.with_suffix('.json'),receipt);put(OUT/'latest_checkpoint.json',receipt)
 return receipt
def tests(device):
 setup();verify();torch.set_default_dtype(torch.float32);started=time.monotonic();p=read(OUT/'frozen_protocol.json')['profile'];a,l,init=build(p,device)
 baseline=torch.load(OUT/'reference/vsoed.pt',map_location='cpu',weights_only=False);assert baseline['profile']==p and baseline['horizon']==30 and baseline['implementation']=='upstream_GMM_I'
 dense_native,_,_=build(p,'cpu',candidate=False);dense_native.actor_net.load_state_dict(baseline['actor_state'],strict=True);del dense_native,baseline
 x=torch.randn(4,3,256,device=device,requires_grad=True);b=a.actor_net.net[4];actual=b(x);phead=b.gate(x).softmax(-1);explicit=sum(h(x)*phead[...,k,None] for k,h in enumerate(b.heads));torch.testing.assert_close(actual,explicit)
 assert actual.shape==(4,3,2) and torch.isfinite(actual).all();torch.testing.assert_close(phead,torch.full_like(phead,.25))
 g1=torch.autograd.grad(actual.square().sum(),tuple(b.parameters())+(x,),retain_graph=True);g2=torch.autograd.grad(explicit.square().sum(),tuple(b.parameters())+(x,))
 for g,h in zip(g1,g2):torch.testing.assert_close(g,h,rtol=5e-4,atol=2e-5)
 assert sum(v.numel() for v in a.actor_net.parameters())==162304
 assert 2*sum(m.weight.numel() for m in a.actor_net.modules() if isinstance(m,nn.Linear))==323056
 with torch.device(device):
  wrapper=UpstreamActor(a.actor_net,30)
  for t in [0,29]:
   ds=torch.randn(4,t,2);ys=torch.randn(4,t,1);features=torch.zeros(4,121,dtype=torch.float64);slots=features[:,1:].reshape(4,30,4);slots[:,:t,:2]=ds.double()/4;slots[:,:t,2]=ys[:,:,0].double()/3
   native=a.form_actor_input(t,ds,ys);torch.testing.assert_close(wrapper(features,t),a.actor_net(native).double())
  y=a.actor_net(torch.randn(4,3,117));assert y.shape==(4,3,2) and y.abs().max()<=4
  loss=-a.critic_net(torch.cat([torch.randn(4,3,117),y],-1)).mean();loss.backward()
  assert b.gate.weight.grad is not None and b.gate.weight.grad.norm()>0
  assert all(sum(float(p.grad.norm()) for p in h.parameters())>0 for h in b.heads)
  assert a.actor_net.net[0].weight.grad.norm()>0 and a.actor_net.net[2].weight.grad.norm()>0
  assert all(torch.isfinite(v.grad).all() for v in a.actor_net.parameters() if v.grad is not None)
 # Complete tiny native continuation fixture, same algorithms, reduced sampling only for tests.
 tiny=dict(p,updates=1,new_trajectories=4,replay_batch=4,initial_replay=8,replay_capacity=16,posterior_updates=1,critic_updates=1,log_every=1)
 target_initial=[v.detach().clone() for v in a.actor_target_net.parameters()]
 attach(a,l);train_native(a,l,tiny)
 for old,online,target in zip(target_initial,a.actor_net.parameters(),a.actor_target_net.parameters()):torch.testing.assert_close(target,old*.9+online*.1)
 path=OUT/'checkpoints'/('disposable_'+device+'.pkl');saved_rng=rng()
 with open(path,'wb') as f:cloudpickle.dump(dict(algorithm=a,logger=l,rng=saved_rng),f)
 with open(path,'rb') as f:restored=cloudpickle.load(f)
 aa=restored['algorithm'];assert digest(a.actor_net.state_dict())==digest(aa.actor_net.state_dict()) and aa.update==1
 assert torch.equal(a._runtime_buffer['params'],aa._runtime_buffer['params'])
 assert aa.actor_optimizer.state_dict()['param_groups']==a.actor_optimizer.state_dict()['param_groups']
 restore(saved_rng);train_native(a,l,tiny,target=2);expected=digest(a.actor_net.state_dict());restore(restored['rng']);train_native(aa,restored['logger'],tiny,target=2);assert expected==digest(aa.actor_net.state_dict()),'native resume mismatch'
 # Independent evaluator adapter smoke: no final bank, no extra performance pilot.
 from src.research.engine import rollout
 from src.research.benchmarks import make_benchmark
 from src.research.location_evaluation import contrast_scores
 bank=np.load(read(OUT/'frozen_protocol.json')['bank']);bench=make_benchmark(read(OUT/'frozen_protocol.json')['physical'])
 with torch.no_grad(),torch.device(device):
  data=rollout(bench,UpstreamActor(a.actor_net,30),30,2,2026110205150101,0,truth=torch.as_tensor(bank['truth'][:2],device=device),noise=torch.as_tensor(bank['noise'][:2],device=device))
  score,_=contrast_scores(bench,data,torch.as_tensor(bank['contrasts'][:16]),16,device);assert score.shape==(2,30) and torch.isfinite(score).all()
 record=dict(status='PASS',device=device,seconds=time.monotonic()-started,checks=['authoritative baseline checkpoint profile and strict native topology','shape2D3D','explicit soft head mixture values and gradients','uniform gate simplex','252-neuron fresh initialization identity','all four heads execute','analytic parameter/FLOP counts','physical output once','t0/t29 native history parity','helper/RNG preservation','target independent copy and native Polyak','optimizer coverage','critic action gradient to gate/experts','complete replay/Adam/scheduler/RNG checkpoint continuation bitwise','independent evaluator adapter smoke'],init=init)
 put(OUT/('tests_'+device+'.json'),record);print(json.dumps(record),flush=True)
def train():
 global STOP
 setup();verify();torch.set_default_dtype(torch.float32);p=read(OUT/'frozen_protocol.json')['profile'];start=time.monotonic()
 signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True))
 assert not (OUT/'latest_checkpoint.json').exists(),'No automatic restart/resume'
 a,l,init=build(p,'cuda');l.smoke=False;l.start=start;attach(a,l)
 put(OUT/'initialization.json',dict(init=init,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),actor_parameters=162304,critic_parameters=sum(v.numel() for v in a.critic_net.parameters()),posterior_parameters=sum(v.numel() for nets in a.post_approx.poi_post_nets.values() for n in nets for v in n.parameters()),source_sha256=sha(OUT/'source_manifest.json')))
 status='COMPLETE'
 try:train_native(a,l,p)
 except SafeStop:status='INCOMPLETE_TIME_CAP' if time.monotonic()-start>=41400 else 'STOPPED_SAFELY'
 cp=save_runtime(l,OUT/'checkpoints'/f'update{a.update}_final_runtime.pkl')
 actorpath=OUT/'checkpoints'/f'update{a.update}_actor.pt';tmp=actorpath.with_suffix('.tmp');torch.save(dict(method='direct_head_moe_vsoed_seed101',actor_state=a.actor_net.state_dict(),profile=p,successful_iterations=int(a.update),init=init,source_sha256=sha(OUT/'source_manifest.json')),tmp);tmp.replace(actorpath)
 result=dict(status=status,successful_iterations=int(a.update),native_optimizer_updates=l.counts,fresh_primary_trajectories=p['initial_replay']+a.update*p['new_trajectories'],charged_training_seconds=time.monotonic()-start,checkpoint=cp,actor_checkpoint=str(actorpath),actor_sha256=sha(actorpath),native_schedule_complete=a.update==10001)
 put(OUT/'training_result.json',result);put(OUT/'status.json',result);assert status!='COMPLETE' or (a.update==10001 and l.counts==dict(actor=10001,critic=50005,posterior=1500150))
 print(json.dumps(result),flush=True)
def evaluate():
 setup();verify();started=time.monotonic();r=read(OUT/'training_result.json');assert r['status']=='COMPLETE' and r['native_schedule_complete'];assert sha(r['actor_checkpoint'])==r['actor_sha256']
 from src.research.benchmarks import make_benchmark
 from src.research.engine import rollout
 from src.research.location_evaluation import contrast_scores
 protocol=read(OUT/'frozen_protocol.json');assert sha(protocol['bank'])==protocol['bank_sha256'];bank=np.load(protocol['bank']);assert hashlib.sha256(bank['truth'].tobytes()+bank['noise'].tobytes()+bank['contrasts'].tobytes()).hexdigest()==protocol['bank_identity']
 torch.set_default_dtype(torch.float32);a,l,init=build(protocol['profile'],'cuda');cp=torch.load(r['actor_checkpoint'],map_location='cuda',weights_only=False);a.actor_net.load_state_dict(cp['actor_state'],strict=True);a.actor_net.eval();actor=UpstreamActor(a.actor_net,30)
 benchmark=make_benchmark(protocol['physical']);scores=[]
 with torch.no_grad(),torch.device('cuda'):
  for start in range(0,2000,16):
   stop=min(start+16,2000);truth=torch.as_tensor(bank['truth'][start:stop],device='cuda');noise=torch.as_tensor(bank['noise'][start:stop],device='cuda')
   data=rollout(benchmark,actor,30,stop-start,2026110205150101+start*97451,0,truth=truth,noise=noise);info,_=contrast_scores(benchmark,data,torch.as_tensor(bank['contrasts']),16384,'cuda');scores.extend(info[:,-1].numpy())
 with torch.no_grad(),torch.device('cuda'):
  probe=torch.zeros(16,117);probe[:,0]=1
  for _ in range(10):a.actor_net(probe)
  torch.cuda.synchronize();latstart=time.perf_counter()
  for _ in range(100):a.actor_net(probe)
  torch.cuda.synchronize();latency=(time.perf_counter()-latstart)/100
 put(OUT/'evaluation/inference_latency.json',dict(seconds_per_fixed_batch16=latency,warmup10=True,repeats=100,actor_only=True,online_search=False,gpu=torch.cuda.get_device_name()))
 values=np.asarray(scores);np.save(OUT/'evaluation/candidate_per_case.npy',values)
 def stats(v):m=float(v.mean());s=float(v.std(ddof=1)/math.sqrt(len(v)));return dict(mean=m,mcse=s,ci_low=m-1.96*s,ci_high=m+1.96*s)
 rows=[dict(method='direct_head_moe_vsoed',**stats(values))]
 for name in ['vsoed','sparse_moe_vsoed','rl_sboed','scorebed','dad','fixed','random','v1','hierarchical']:
  path=OUT/'reference/baselines'/f'development_{name}_seed101.npy' if name not in ['v1','hierarchical','sparse_moe_vsoed'] else OUT/'reference'/({'v1':'v1_scores.npy','hierarchical':'hierarchical_scores.npy','sparse_moe_vsoed':'sparse_scores.npy'}[name])
  if not path.exists():rows.append(dict(method=name,status='UNAVAILABLE'));continue
  ref=np.load(path);d=values-ref;np.save(OUT/'evaluation'/f'candidate_minus_{name}.npy',d);paired=stats(d);rows.append(dict(method=name,**stats(ref),paired_difference=paired['mean'],paired_mcse=paired['mcse'],paired_ci_low=paired['ci_low'],paired_ci_high=paired['ci_high']))
 assert sha(r['actor_checkpoint'])==r['actor_sha256'];put(OUT/'paired_endpoint_results.json',dict(rows=rows,primary='direct-head minus native vsOED',secondary='all other rows',actor_parameters=162304,seconds=time.monotonic()-started));put(OUT/'evaluation/paired_baseline_comparison.json',dict(rows=rows,bank_sha256=protocol['bank_sha256'],checkpoint_unchanged=True,seconds=time.monotonic()-started,seed_replication=False,final_bank_access=False))
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with open(OUT/'baseline_comparison.csv','w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 (OUT/'RESULTS.md').write_text('# Completed development candidate\n\n'+json.dumps(dict(training=r,comparison=rows,evaluation_seconds=time.monotonic()-started),indent=2)+'\n\nOne seed; finite prior-sPCE conditional on reused development bank. No novelty, exact MI or across-seed superiority established. Native vsOED is primary comparison; secondary comparisons exploratory. Architecture parameters and native work approximately matched to vsOED, hardware/active dispatch/wall cost differ. No follow-up training or promotion.\n')
 put(OUT/'status.json',dict(status='TRAINING_AND_EVALUATION_COMPLETE',comparison=rows));print(json.dumps(rows),flush=True)
def supervise():
 setup();held=acquire();controller=(OUT/'controller.lock').open('a+');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
 try:
  assert read(OUT/'tests_cpu.json')['status']=='PASS'
  phases=[('tests_gpu',['tests','cuda'],'tests.log'),('train',['train'],'logs/train_seed101.log'),('evaluate',['evaluate'],'logs/evaluate.log')]
  for phase,args,log in phases:
   if phase=='evaluate' and read(OUT/'training_result.json')['status']!='COMPLETE':break
   with open(OUT/log,'a') as f:child=subprocess.Popen([sys.executable,str(__file__)]+args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   put(OUT/'process.json',dict(supervisor_pid=os.getpid(),worker_pid=child.pid,phase=phase,command=[sys.executable,str(__file__)]+args,endpoint_evaluation_dependency='native COMPLETE10001 then automatic evaluator',start_epoch=time.time(),hard_cap_seconds=42300))
   while child.poll() is None:
    elapsed=time.monotonic()-started
    if elapsed>=41400:(OUT/'STOP_REQUESTED').touch()
    if elapsed>=42300:
     # Child handle, UID, process group and command verified immediately before signaling.
     proc=pathlib.Path('/proc')/str(child.pid)
     if proc.exists() and str(__file__).encode() in (proc/'cmdline').read_bytes() and proc.stat().st_uid==os.getuid() and os.getpgid(child.pid)==child.pid:os.killpg(child.pid,signal.SIGKILL)
     child.wait();raise RuntimeError('Verified worker group hard cutoff at11h45m')
    time.sleep(2)
   if child.returncode:raise RuntimeError(f'{phase} failed rc{child.returncode}; no automatic restart')
  put(OUT/'supervisor_final.json',dict(status='EXITED',seconds=time.monotonic()-started))
 finally:
  for f in held:f.close()
  controller.close();(OUT/'ARTIFACT_INDEX.txt').write_text('\n'.join(str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()))
if __name__=='__main__':
 try:
  if sys.argv[1]=='tests':tests(sys.argv[2])
  else:globals()[sys.argv[1]]()
 except BaseException as exc:
  put(OUT/('failure_'+sys.argv[1]+'.json'),dict(error=str(exc),traceback=traceback.format_exc()));raise
