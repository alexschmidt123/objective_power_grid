"""New V5:offline native-posterior snapshot refresh;unchanged native actor-critic objective."""
from pathlib import Path
import argparse,copy,fcntl,importlib.util,io,json,os,subprocess,sys,time
M=Path.home()/'Documents/objective_power_grid';HERE=M/'src/research/moe_sboed_v5_belief_20261119';O=M/'experiments/location_finding/moe_sboed_v5_belief_20261119';R=M/'reports/moe_sboed_v5_belief_20261119';C=M/'configs/moe_sboed_v5_belief_20261119.json'
spec=importlib.util.spec_from_file_location('new_v5_offline_diagnostics',M/'tools/moe_sboed_v5_belief_diagnostics_20261119.py');d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d);p=d.p
sys.path.insert(0,str(HERE));from belief_policy import V5Actor
import torch,numpy as np
NAMES={'learned':'MoE-sBOED V5','uniform':'MoE-sBOED V5-uniform'}
def read(path):return json.loads(Path(path).read_text())
def put(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(path,value)
def cfg():return read(C)
def identity():return {str(f.relative_to(M)):p.sha(f) for f in [*HERE.glob('*.py'),C,M/'tools/moe_sboed_v5_belief_diagnostics_20261119.py',M/'tools/moe_sboed_v5_belief_20261119.py',* [M/name for name in cfg()['frozen_history_sources']]]}
def actor(parent,arm):
 model=p.new_actor(parent,arm);model.__class__=V5Actor;return model

def refresh(a,update):
 """Called only by offline train_worker BETWEEN native blocks. No gradients/optimizer/episodes."""
 assert not torch.is_grad_enabled(),'Offline snapshot transfer uses no_grad'
 before={name:value.detach().clone() for name,value in a.actor_net.state_dict().items() if not name.startswith('query.inference.belief.')}
 old=p.n.digest(a.actor_net.query.inference.belief.state_dict());ids=[id(v) for v in a.actor_net.parameters()]
 snapshot=type(a.actor_net.query.inference.belief)(a.post_approx).to(next(a.actor_net.parameters()).device);snapshot.eval();snapshot.requires_grad_(False)
 assert sum(v.numel() for v in snapshot.parameters())==0
 a.actor_net.query.inference.belief=snapshot
 a.actor_target_net.query.inference.belief=copy.deepcopy(snapshot)
 assert ids==[id(v) for v in a.actor_net.parameters()]
 for name,value in before.items():torch.testing.assert_close(a.actor_net.state_dict()[name],value,atol=0,rtol=0)
 current=p.n.digest(snapshot.state_dict());assert current==p.n.digest(a.actor_target_net.query.inference.belief.state_dict())
 return dict(continuation_update=update,native_update=int(a.update),old_belief_digest=old,new_belief_digest=current,only_belief_buffers_changed=True,
  target_belief_identical=True,optimizer_parameter_identities_preserved=True,new_actor_optimizer_updates=0,new_posterior_optimizer_updates=0,new_simulator_episodes=0)

def audit():
 assert read(R/'PREFLIGHT_RESULTS.json')['pilot_gate']
 torch.set_num_threads(2);state=p.load_parent('cpu');a=state['algorithm'];base=p.new_actor(a.actor_net,'learned');aa=actor(a.actor_net,'learned');bb=actor(a.actor_net,'uniform')
 assert p.n.digest(base.state_dict())==p.n.digest(aa.state_dict())==p.n.digest(bb.state_dict())
 data=np.load(O/'development_histories.npz');lib=p.n.load_library();ds=torch.tensor(data['designs'][:2]);ys=torch.tensor(data['observations'][:2]);x=torch.cat([lib.packed(ds,ys,t) for t in [0,5,15,25,29]])
 with torch.no_grad():torch.testing.assert_close(aa(x),bb(x),atol=0,rtol=0);torch.testing.assert_close(aa(x),base(x),atol=0,rtol=0)
 a.actor_net=aa;a.actor_target_net=copy.deepcopy(aa);p.engine.optimizer(a)
 with torch.no_grad():receipt=refresh(a,0)
 a.actor_net.configure('joint');a.critic_net.requires_grad_(False);(-a.critic_net(torch.cat([x,a.actor_net(x)],-1)).mean()).backward();grads=p.gradients(a.actor_net)
 assert all(grads['experts.'+str(k)]>0 for k in range(4)) and grads['router.2']>0
 deployment=d.deployment_audit(a.actor_net,x);put(R/'NEW_V5_ARCHITECTURE_AUDIT.json',dict(status='PASS',paired_initialization_exact=True,initial_actions_equal_to_previous_V5=True,refresh=receipt,gradient_norms=grads,parameter_counts=p.parameter_counts(a.actor_net),deployment=deployment))
 print('NEW_V5_AUDIT_PASS',flush=True)

def prepare():
 assert not C.exists();assert read(R/'PREFLIGHT_RESULTS.json')['pilot_gate'] and read(R/'NEW_V5_ARCHITECTURE_AUDIT.json')['status']=='PASS'
 history=read(p.O/'SOURCE_MANIFEST.json')
 for name,value in history.items():assert p.sha(M/name)==value
 birth=int(subprocess.check_output(['stat','-c','%W',str(O)],text=True));assert birth>0
 credit=max(0,read(R/'PREFLIGHT_RESULTS.json')['seconds'],(R/'PREFLIGHT_RESULTS.json').stat().st_mtime-birth)+5
 config=dict(study='ONE_MATCHED_NEW_V5_NATIVE_BELIEF_REFRESH_SEED101',seed=101,methods=NAMES,
  only_change='Offline refresh actor/target frozen belief from their OWN native posterior at continuation updates200,400,...1800. No refresh after terminal2000.',
  architecture='History-conditioned previous V5 unchanged:four128 experts,weighted sum128,original644->64->4 router,128->32->2 head,bounded.2 weak branch.',
  initialization='Original V4 protocol/frozen V2 parent16001;never trained previousV5 runtime as initialization. Diagnostic runtime is read-only evidence only.',
  native_objective='discount1,use_PCE=False,unchanged native variational actor-critic EIG;no new rewards/losses/optimization updates',
  warmup_updates=200,joint_updates=1800,actor_updates=2000,critic_updates=10000,posterior_updates=300000,new_episodes=2000000,observations=60000000,
  posterior_refresh_updates=list(range(200,2000,200)),hard_seconds_per_run=14400,preparation_seconds_conservatively_charged=credit,
  no_online_optimization=True,no_test_time_posterior_updates=True,diagnostic_seed_range_fix='Native Torch/legacyNumpy seeding modulo2**32;PCG64 development seed unchanged;failed attempt preserved',
  reference_limitations='9/12 importance cases ESS>=256;finite reference not exact or guaranteed multimode coverage. Critic rank matches2/12 by noisy mean,not proof of a bug.',
  critic_change='NONE:diagnostic does not justify substituting critic objective. Only native posterior snapshot mechanism changes.',
  evaluation=p.engine.LOCK,canonical_test_selection=False,checkpoint='Fixed terminal2000;no canonical tuning',
  uniform='Exactly.25,router bypassed/frozen throughout training,target and evaluation;all other components get matching native training and refresh schedule',
  expansion='No automatic additional seeds,no broad search',frozen_history_sources=history)
 put(C,config);put(R/'TRAINING_PROTOCOL.json',config);put(O/'NEW_SOURCE_MANIFEST.json',identity())
 print('PREPARATION_CREDIT_SECONDS',credit,flush=True)

def init_receipt(a):
 return dict(actor=p.n.digest(a.actor_net.state_dict()),actor_target=p.n.digest(a.actor_target_net.state_dict()),critic=p.n.digest(a.critic_net.state_dict()),critic_target=p.n.digest(a.critic_target_net.state_dict()),
  posterior=p.tree_digest({k:[net.state_dict() for net in nets] for k,nets in a.post_approx.poi_post_nets.items()}),optimizers=p.tree_digest([op.state_dict() for op,_ in p.engine.opts(a)]),replay=p.tree_digest(a._runtime_buffer),rng=p.tree_digest(p.n.original.rng()))

def train_worker(arm):
 start=time.monotonic();credit=cfg()['preparation_seconds_conservatively_charged'];out=O/arm;out.mkdir(exist_ok=False);p.setup();assert identity()==read(O/'NEW_SOURCE_MANIFEST.json')
 with p.lease(),p.engine.Resources(out):
  try:
   state=p.load_parent('cuda');a=state['algorithm'];model=actor(a.actor_net,arm);a.actor_net=model;a.actor_target_net=copy.deepcopy(model);a.actor_target_net.configure('frozen');p.engine.optimizer(a);p.n.original.restore(state['rng']);p.engine.seed_all(101)
   put(out/'INITIALIZATION.json',init_receipt(a));put(out/'CONFIG.json',dict(method=NAMES[arm],protocol=cfg(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip(),source_manifest=identity()))
   phases=[];refreshes=[];p.COST.clear();trainstart=time.monotonic()
   for block in range(10):
    phase='warmup' if block==0 else 'joint';folder=out/f'block{block:02d}_{phase}';folder.mkdir()
    phases.append(p.engine.train_phase(a,folder,phase,200,start-credit,start+12600-credit))
    put(out/'PROGRESS.json',dict(method=NAMES[arm],seed=101,completed_updates=200*(block+1),target=2000,counts={k:sum(v['counts'][k] for v in phases) for k in ['actor','critic','posterior']}))
    if block<9:
     with torch.no_grad():refreshes.append(refresh(a,200*(block+1)))
     put(out/'BELIEF_REFRESHES.json',refreshes)
   trainseconds=time.monotonic()-trainstart;counts={k:sum(v['counts'][k] for v in phases) for k in ['actor','critic','posterior']};assert counts==dict(actor=2000,critic=10000,posterior=300000)
   assert sum(v['new_episodes'] for v in phases)==2000000
   # Final policy is trained200 updates under the update1800 snapshot;no untrained final refresh.
   torch.save(dict(actor_state=model.state_dict(),native_iteration=int(a.update),belief_snapshot_update=1800),out/'final_actor.pt')
   with (out/'runtime.pkl').open('wb') as f:p.n.cloudpickle.dump(dict(algorithm=a,rng=p.n.original.rng(),method=NAMES[arm],seed=101),f,protocol=5)
   model.configure('frozen');model.eval();native_calls=dict(p.COST);lib=p.n.load_library();data=np.load(O/'development_histories.npz');xx=lib.packed(torch.tensor(data['designs'][:2],device='cuda'),torch.tensor(data['observations'][:2],device='cuda'),15)
   deployment=d.deployment_audit(model,xx);put(out/'NO_ONLINE_OPTIMIZATION_AUDIT.json',deployment)
   evout=out/'canonical_evaluation';evout.mkdir()
   with torch.no_grad():values,ev=p.engine.evaluate(model,evout,arm,start+14100-credit,collect=False)
   total=time.monotonic()-start+credit;assert total<=14400 and identity()==read(O/'NEW_SOURCE_MANIFEST.json')
   put(out/'RESULTS.json',dict(status='COMPLETE',method=NAMES[arm],seed=101,phases=phases,counts=counts,actor_updates=2000,new_episodes=2000000,new_observations=60000000,
    training_seconds=trainseconds,logical_total_seconds=total,preparation_credit_seconds=credit,refreshes=refreshes,training_model_queries=native_calls,evaluation=ev,checkpoint=str(out/'final_actor.pt'),checkpoint_sha256=p.sha(out/'final_actor.pt')))
  except Exception as error:
   if 'model' in locals():torch.save(dict(actor_state=model.state_dict(),incomplete=True,error=repr(error)),out/'incomplete_actor.pt')
   put(out/'STATUS.json',dict(status='FAILED_PRESERVED',error=repr(error)));raise

def report():
 rows={};vectors={};comparisons={}
 previous=read(M/'reports/moe_sboed_v5_history_canonical_20261119/RESULTS.json')
 for arm in NAMES:
  path=O/arm/'RESULTS.json'
  if path.exists():
   r=read(path);assert r['evaluation']['protocol']==p.engine.LOCK;v=np.load(O/arm/'canonical_evaluation'/(arm+'_scores.npy'));vectors[arm]=v;rows[arm]=dict(method=NAMES[arm],**p.engine.summaries(v))
 for key in ['previous_learned','previous_uniform','vsoed','dad']:
  r=previous['rows'][key];assert p.sha(r['checkpoint'])==r['checkpoint_sha256'];v=np.load(r['scores']);vectors[key]=v;rows[key]=dict(method=r['method'],**p.engine.summaries(v))
 if 'learned' in vectors:
  for key in ['uniform','previous_learned','previous_uniform','vsoed']:
   if key in vectors:comparisons['newV5_minus_'+key]=p.engine.summaries(vectors['learned']-vectors[key])
 if 'uniform' in vectors:comparisons['uniform_minus_V4uniform']=p.engine.summaries(vectors['uniform']-vectors['previous_uniform'])
 if len([x for x in NAMES if x in vectors])==2:assert read(O/'learned/INITIALIZATION.json')==read(O/'uniform/INITIALIZATION.json')
 status='COMPLETE' if all(k in vectors for k in NAMES) else 'PENDING'
 put(R/'RESULTS.json',dict(status=status,rows=rows,comparisons=comparisons,evaluation=p.engine.LOCK,automatic_expansion=False,one_continuation_seed=True))
 text='# New V5:offline native-belief refresh,seed101\n\nStatus:'+status+'. ORIGINAL canonical V2/V4/DAD/vsOED bank,no retraining controls.\n\n|Method|EIG lower bound|MCSE|\n|---|---:|---:|\n'
 for k in ['learned','uniform','previous_learned','previous_uniform','vsoed']:
  if k in rows:r=rows[k];text+=f"|{r['method']}|{r['mean']:.5f}|{r['MCSE']:.5f}|\n"
  else:text+=f'|{NAMES[k]}|PENDING|PENDING|\n'
 text+='\n|Paired comparison|Difference|95% case evaluation CI|\n|---|---:|---|\n'
 for key,r in comparisons.items():text+=f"|{key}|{r['mean']:+.5f}|[{r['CI95'][0]:+.5f},{r['CI95'][1]:+.5f}]|\n"
 text+='\nOne continuation seed,not training-seed robustness. Finite1M-contrast lower bounds;MCSE excludes contrast bias. No automatic expansion or canonical-bank tuning.\n';(R/'RESULTS.md').write_text(text)

def controller():
 handle=(O/'.new_v5_controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (O/'NEW_V5_LAUNCH.json').exists();put(O/'NEW_V5_LAUNCH.json',dict(pid=os.getpid(),source=identity()))
 done=[]
 for arm in NAMES:
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():put(O/'NEW_V5_STATUS.json',dict(status='WAITING_FOR_GPU',next=arm));time.sleep(20)
  put(O/'NEW_V5_STATUS.json',dict(status='RUNNING',current=arm,completed=done,pending=[x for x in NAMES if x not in done and x!=arm]))
  seconds=int(14370-cfg()['preparation_seconds_conservatively_charged'])
  with (O/(arm+'.log')).open('x') as log:
   child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(seconds),sys.executable,'-B','-u',str(M/'tools/moe_sboed_v5_belief_20261119.py'),'--task','train','--arm',arm],stdout=log,stderr=subprocess.STDOUT,cwd=M)
   put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),worker_pid=child.pid,method=NAMES[arm],seed=101));code=child.wait()
  if code:put(O/'NEW_V5_STATUS.json',dict(status='FAILED_PRESERVED',current=arm,returncode=code));report();return
  done.append(arm);report()
 put(O/'NEW_V5_STATUS.json',dict(status='COMPLETE',completed=done,no_expansion=True))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['audit','prepare','train','controller','report'],required=True);parser.add_argument('--arm',choices=list(NAMES),default='learned');args=parser.parse_args()
 if args.task=='audit':audit()
 elif args.task=='prepare':prepare()
 elif args.task=='train':train_worker(args.arm)
 elif args.task=='controller':controller()
 else:report()
if __name__=='__main__':main()
