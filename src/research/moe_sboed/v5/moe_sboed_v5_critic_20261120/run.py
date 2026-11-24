"""One seed101 pilot:previous V5 actor unchanged,belief-aware native residual critic."""
from pathlib import Path
import argparse,copy,fcntl,importlib.util,json,os,subprocess,sys,time
M=Path.home()/'Documents/objective_power_grid';HERE=M/'src/research/moe_sboed_v5_critic_20261120';O=M/'experiments/location_finding/moe_sboed_v5_critic_20261120';R=M/'reports/moe_sboed_v5_critic_20261120';C=M/'configs/moe_sboed_v5_critic_20261120.json'
spec=importlib.util.spec_from_file_location('critic_previous_v5',M/'src/research/moe_sboed_v5_history_20261119/run.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
sys.path.insert(0,str(HERE));from critic import BeliefCritic,install
import torch,numpy as np
ORIGINAL_NEW_ACTOR=p.new_actor;ORIGINAL_LOAD_PARENT=p.load_parent
NAMES={'learned':'MoE-sBOED V5','uniform':'MoE-sBOED V5-uniform'}
def read(path):return json.loads(Path(path).read_text())
def put(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(path,value)
def cfg():return read(C)
def source_identity():return {str(f.relative_to(M)):p.sha(f) for f in [*HERE.glob('*.py'),C,M/'tools/moe_sboed_v5_critic_20261120.py',* [M/path for path in cfg()['frozen_previous_sources']]]}

def build(arm,device):
 state=ORIGINAL_LOAD_PARENT(device);a=state['algorithm'];actor=ORIGINAL_NEW_ACTOR(a.actor_net,arm)
 a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor);a.actor_target_net.configure('frozen');p.engine.optimizer(a)
 install(a,actor.query.inference.belief,cfg()['critic_initialization_seed'])
 if device=='cuda':p.n.original.restore(state['rng'])
 p.engine.seed_all(101);return state,a

def bind():
 # Parameterize the immutable previous runner in memory;never edit its source/configs.
 p.O=O;p.R=R;p.C=C;p.cfg=cfg;p.build=build;p.source_identity=source_identity
 p.bank=lambda:p.engine.canonical_load()[0]

def prepare():
 assert not O.exists() and not R.exists() and not C.exists();O.mkdir(parents=True);R.mkdir(parents=True)
 old=read(M/'configs/moe_sboed_v5_history_20261119.json');manifest=read(M/'experiments/location_finding/moe_sboed_v5_history_20261119/SOURCE_MANIFEST.json')
 for path,value in manifest.items():assert p.sha(M/path)==value
 old.update(study='ONE_MATCHED_PREVIOUS_V5_ACTOR_BELIEF_CRITIC_SEED101',methods=list(NAMES.values()),
  change='Training-only Q=original trainable critic+zero-output residual on[observed_history117,action2,frozen_belief_summary14]. Actor is previous historyV5,unchanged.',
  critic_architecture='Inherited119->256ReLU->256ReLU->256ReLU->1 plus133->128SiLU->64SiLU->1 zero-output residual.',
  critic_initialization_seed=202611207101,frozen_previous_sources=manifest,
  optimizer='Preserve inherited base-critic Adam state;add residual group at identical current LR and gamma. Native TD loss/targets/updates unchanged.',
  caching='Exact value-checked last-prefix memoization of fixed belief summaries;no scientific approximation. Cache excluded from runtime serialization.',
  inference='NO online optimization;critic not called by deployed actor. Frozen actor belief retained;retired belief-refresh is not used.',
  evaluation={**old['evaluation'],**p.engine.LOCK,'bank':'Exact original canonical V2/V4/DAD/vsOED bank;no replacement bank'},
  expansion='One matched seed101 pilot only,no automatic seeds102-105. No canonical tuning.',
  architectural_caveat='Changes critic capacity and explicit belief input jointly;cannot attribute any improvement separately.',
  preflight='CPU correctness/native update fixture and small production-budget GPU timing test;preparation costs reported and charged to both caps.')
 old.pop('saturation_sensitivity',None);old.pop('history_limits',None);old.pop('retirement',None);old['evaluation'].pop('seed',None)
 old['architecture']='Previous history-conditioned V5 actor unchanged;training-only belief-aware native residual critic'
 put(C,old);put(R/'PROTOCOL.json',old);put(O/'SOURCE_MANIFEST.json',source_identity())
 print('PREPARED',flush=True)

def audit():
 torch.set_num_threads(2);state,a=build('learned','cpu');actor=a.actor_net
 fixtures=np.load(M/'experiments/location_finding/moe_sboed_v5_history_20261119/development_histories.npz') if (M/'experiments/location_finding/moe_sboed_v5_history_20261119/development_histories.npz').exists() else np.load(M/'experiments/location_finding/moe_sboed_v5_belief_20261119/development_histories.npz')
 ds=torch.tensor(fixtures['designs'][:2]);ys=torch.tensor(fixtures['observations'][:2]);lib=p.n.load_library();x=torch.cat([lib.packed(ds,ys,t) for t in [0,5,15,25,29]]);xx=torch.cat([x,actor(x).detach()],-1).requires_grad_(True)
 value=a.critic_net(xx);base=a.critic_net.base(xx);torch.testing.assert_close(value,base,atol=0,rtol=0)
 got=torch.autograd.grad(value.sum(),xx,retain_graph=True)[0][:,-2:];expected=torch.autograd.grad(base.sum(),xx)[0][:,-2:];torch.testing.assert_close(got,expected,atol=0,rtol=0)
 before=a.critic_net.context_evaluations;a.critic_net(xx.detach());assert a.critic_net.context_evaluations==before
 changed=xx.detach().clone();changed[:,88]+=.5;a.critic_net(changed);assert a.critic_net.context_evaluations>before
 initial_actor=p.n.digest(actor.state_dict());old_init=read(M/'experiments/location_finding/moe_sboed_v5_history_20261119/learned/INITIALIZATION.json');assert initial_actor==old_init['actor_digest']
 # Residual learns with the native critic optimizer;no new loss is used in formal training.
 for step in range(2):
  a.critic_optimizer.zero_grad();loss=(a.critic_net(xx.detach())-torch.linspace(-1,1,len(xx))[:,None]).square().mean();loss.backward()
  assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in a.critic_net.residual.parameters())
  if step==1:assert a.critic_net.residual[0].weight.grad.norm()>0
  a.critic_optimizer.step()
 actor.configure('joint');actor.zero_grad();(-a.critic_net(torch.cat([x,actor(x)],-1)).mean()).backward();grad=p.gradients(actor);assert grad['router.2']>0 and all(grad[f'experts.{k}']>0 for k in range(4))
 spec=importlib.util.spec_from_file_location('critic_no_online_guard',M/'tools/moe_sboed_v5_belief_diagnostics_20261119.py');d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
 deployment=d.deployment_audit(actor,x)
 uniform=ORIGINAL_NEW_ACTOR(ORIGINAL_LOAD_PARENT('cpu')['algorithm'].actor_net,'uniform');assert p.n.digest(uniform.state_dict())==initial_actor
 put(R/'AUDIT.json',dict(status='PASS',actor_initial_digest=initial_actor,actor_identical_to_previous_V5_initialization=True,paired_initial_actor_identical=True,
  initial_Q_and_design_gradients_bitwise_equal_to_parent=True,cache_reuses_equal_prefix_and_invalidates_changed_history=True,residual_gradients_finite=True,expert_router_gradients=grad,
  deployment=deployment,critic_parameters=sum(v.numel() for v in a.critic_net.parameters()),residual_parameters=sum(v.numel() for v in a.critic_net.residual.parameters()),
  native_objective='Formal training uses unmodified native actor/critic/posterior updates;synthetic disposable gradient fixture is not training.',formal_training_updates=0))
 print('AUDIT_PASS',flush=True)

def smoke():
 p.setup();out=O/'native_smoke';out.mkdir()
 with p.lease(),p.engine.Resources(out):
  state,a=build('learned','cuda');start=time.monotonic();phase=p.engine.train_phase(a,out,'warmup',2,start,start+180)
  #2 full native updates,discarded. Formal runs reload original parent and seed.
  put(R/'NATIVE_SMOKE.json',dict(status='PASS',phase=phase,seconds=time.monotonic()-start,discarded=True,
   posterior_context_evaluations=a.critic_net.context_evaluations,cache_hits=a.critic_net.cache_hits,
   actor_checkpoint_not_a_formal_initialization=True))
 print('NATIVE_SMOKE_PASS',flush=True)

def report():
 previous=read(M/'reports/moe_sboed_v5_history_canonical_20261119/RESULTS.json');vectors={};rows={};comparisons={}
 for key,r in previous['rows'].items():
  if key not in ['learned','uniform','previous_learned','previous_uniform','vsoed','dad']:continue
  assert p.sha(r['checkpoint'])==r['checkpoint_sha256'];vectors['reference_'+key]=np.load(r['scores']);rows['reference_'+key]=dict(method=('Previous V5 '+key if key in NAMES else r['method']),**p.engine.summaries(vectors['reference_'+key]))
 for arm in NAMES:
  path=O/arm/'RESULTS.json'
  if path.exists():
   r=read(path);v=np.load(O/arm/'evaluation/scores.npy');vectors[arm]=v;rows[arm]=dict(method=NAMES[arm],**p.engine.summaries(v))
 if 'learned' in vectors:
  for key in ['uniform','reference_learned','reference_previous_learned','reference_vsoed']:
   if key in vectors:comparisons['newV5_minus_'+key]=p.engine.summaries(vectors['learned']-vectors[key])
 if 'uniform' in vectors:comparisons['uniform_minus_previous_uniform']=p.engine.summaries(vectors['uniform']-vectors['reference_uniform'])
 status='COMPLETE' if all(k in vectors for k in NAMES) else 'PENDING';put(R/'RESULTS.json',dict(status=status,rows=rows,comparisons=comparisons,protocol=p.engine.LOCK,automatic_expansion=False))
 text='# New V5:training-only belief-aware native critic,seed101\n\nStatus:'+status+'. Original canonical bank,2000 cases,1M contrasts,T30.\n\n|Method|EIG lower bound|MCSE|\n|---|---:|---:|\n'
 for key in ['learned','uniform','reference_learned','reference_uniform','reference_previous_learned','reference_vsoed']:
  if key in rows:r=rows[key];text+=f"|{r['method']}|{r['mean']:.5f}|{r['MCSE']:.5f}|\n"
  else:text+=f'|{NAMES[key]}|PENDING|PENDING|\n'
 text+='\n|Paired comparison|Difference|95% case CI|\n|---|---:|---|\n'
 for key,r in comparisons.items():text+=f"|{key}|{r['mean']:+.5f}|[{r['CI95'][0]:+.5f},{r['CI95'][1]:+.5f}]|\n"
 text+='\nOne continuation seed,not seed robustness. Finite-contrast lower bounds;MCSE excludes contrast bias. No automatic expansion.\n';(R/'RESULTS.md').write_text(text)

def train(arm):
 bind();p.worker(arm,14400-cfg()['preparation_credit_seconds']);report()

def controller():
 handle=(O/'.controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'LAUNCH.json').exists();assert read(R/'AUDIT.json')['status']=='PASS' and read(R/'NATIVE_SMOKE.json')['status']=='PASS'
 put(O/'LAUNCH.json',dict(pid=os.getpid(),source=source_identity()));done=[]
 for arm in NAMES:
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():put(O/'STATUS.json',dict(status='WAITING_FOR_GPU',next=arm));time.sleep(20)
  put(O/'STATUS.json',dict(status='RUNNING',current=arm,completed=done,pending=[k for k in NAMES if k not in done and k!=arm]))
  with (O/(arm+'.log')).open('x') as log:
   child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(int(14370-cfg()['preparation_credit_seconds'])),sys.executable,'-B','-u',str(M/'tools/moe_sboed_v5_critic_20261120.py'),'--task','train','--arm',arm],stdout=log,stderr=subprocess.STDOUT,cwd=M)
   put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),worker_pid=child.pid,method=NAMES[arm],seed=101));code=child.wait()
  if code:put(O/'STATUS.json',dict(status='FAILED_PRESERVED',arm=arm,returncode=code));report();return
  done.append(arm);report()
 put(O/'STATUS.json',dict(status='COMPLETE',completed=done,automatic_expansion=False))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['prepare','audit','smoke','train','controller','report'],required=True);parser.add_argument('--arm',choices=list(NAMES),default='learned');args=parser.parse_args()
 if args.task=='prepare':prepare()
 elif args.task=='audit':audit()
 elif args.task=='smoke':smoke()
 elif args.task=='train':train(args.arm)
 elif args.task=='controller':controller()
 else:report()
if __name__=='__main__':main()
