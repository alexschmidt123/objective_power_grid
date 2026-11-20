"""One policy-only V5 seed101 pilot;native critic/posterior unchanged."""
from pathlib import Path
import argparse,copy,fcntl,importlib.util,json,os,subprocess,sys,time
M=Path.home()/'Documents/objective_power_grid';HERE=M/'src/research/moe_sboed_v5_history_belief_20261120';O=M/'experiments/location_finding/moe_sboed_v5_history_belief_20261120';R=M/'reports/moe_sboed_v5_history_belief_20261120';C=M/'configs/moe_sboed_v5_history_belief_20261120_train.json'
spec=importlib.util.spec_from_file_location('policy_only_v5_previous',M/'src/research/moe_sboed_v5_history_20261119/run.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
sys.path.insert(0,str(HERE));from expert_policy import Actor
import torch,numpy as np
ORIGINAL_LOAD=p.load_parent
NAMES={'learned':'MoE-sBOED V5','uniform':'MoE-sBOED V5-uniform'}
def read(path):return json.loads(Path(path).read_text())
def put(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(path,value)
def cfg():return read(C)
def source_identity():return {str(f.relative_to(M)):p.sha(f) for f in [*HERE.glob('*.py'),C,M/'tools/moe_sboed_v5_history_belief_20261120.py',* [M/path for path in cfg()['frozen_previous_sources']]]}

def new_actor(parent,arm):
 template=p.StudyMoE(parent,202611150101,'uniform');p.engine.seed_all(202611164101%(2**32));return Actor(template,arm)
def build(arm,device):
 state=ORIGINAL_LOAD(device);a=state['algorithm'];model=new_actor(a.actor_net,arm)
 assert type(a.critic_net).__name__=='Net' and len(a.critic_optimizer.param_groups)==1
 assert p.n.digest(a.critic_net.state_dict())==cfg()['expected_initial_critic_digest']
 assert p.n.digest(model.state_dict())==cfg()['expected_initial_actor_digest']
 a.actor_net=model;a.actor_target_net=copy.deepcopy(model);a.actor_target_net.configure('frozen');p.engine.optimizer(a)
 if device=='cuda':p.n.original.restore(state['rng'])
 p.engine.seed_all(101);return state,a

def bind():
 p.O=O;p.R=R;p.C=C;p.cfg=cfg;p.build=build;p.source_identity=source_identity;p.bank=lambda:p.engine.canonical_load()[0]

def prepare():
 assert not C.exists() and not (O/'LAUNCH.json').exists()
 audit=read(R/'IMPLEMENTATION_AUDIT.json');assert audit['status']=='PASS' and audit['extra_neural_networks']==0 and audit['extra_parameters']==0
 for path,value in audit['source_sha256'].items():assert p.sha(M/path)==value
 old=read(M/'configs/moe_sboed_v5_history_20261119.json');frozen=read(p.O/'SOURCE_MANIFEST.json');initial=read(p.O/'learned/INITIALIZATION.json')
 for path,value in frozen.items():assert p.sha(M/path)==value
 old.update(study='ONE_EXPLORATORY_POLICY_ONLY_HISTORY_BELIEF_MOE_SEED101',methods=list(NAMES.values()),status='LOCKED_BEFORE_GPU_PILOT',
  change='Only role-specific14D history-belief views replace the existing expert conditioner belief inputs;no network/parameter increase.',
  critic='ORIGINAL_NATIVE_UNCHANGED',posterior_training='ORIGINAL_NATIVE_UNCHANGED',extra_neural_networks=0,extra_parameters=0,
  expected_initial_actor_digest=initial['actor_digest'],expected_initial_critic_digest=initial['critic_digest'],frozen_previous_sources=frozen,
  preparation_credit_seconds=180,preparation_credit_scope='Conservative cap reservation,not measured elapsed time;actual CPU/GPU preparation resources reported separately.',
  native_preflight_updates=2,native_preflight_max_seconds=120,
  evaluation={**old['evaluation'],**p.engine.LOCK,'bank':'Exact original canonical V2/V4/DAD/vsOED bank and scorer;no new bank'},
  scientific_questions=['Higher full-horizon EIG from MoE policy with native training fixed','Learned versus independently trained uniform EIG','Distinct and conditionally useful expert computations'],
  checkpoint_selection='Terminal2000 only;no canonical selection or tuning',expansion='No automatic seeds102-105;one exploratory seed101 pair only')
 old['evaluation'].pop('seed',None);old.pop('saturation_sensitivity',None)
 put(C,old);put(R/'TRAINING_PROTOCOL.json',old)
 bank=p.n.BASE/'evaluation/development_validation_bank.npz';assert p.sha(bank)==p.engine.LOCK['bank_sha256']
 alias=O/'development_validation_bank.npz'
 if not alias.exists():alias.symlink_to(bank)
 assert p.sha(alias)==p.engine.LOCK['bank_sha256'];put(O/'BANK_IDENTITY.json',dict(protocol=p.engine.LOCK,source=str(bank),alias=str(alias),new_bank_generated=False))
 put(O/'SOURCE_MANIFEST.json',source_identity());put(R/'SOURCE_MANIFEST.json',source_identity());report();print('LOCKED',flush=True)

def smoke():
 p.setup();out=O/'native_preflight';out.mkdir(exist_ok=False)
 with p.lease(),p.engine.Resources(out):
  state,a=build('learned','cuda');start=time.monotonic();phase=p.engine.train_phase(a,out,'warmup',2,start,start+100)
  put(R/'NATIVE_PREFLIGHT.json',dict(status='PASS',phase=phase,seconds=time.monotonic()-start,discarded=True,actor_updates=2,critic_updates=10,posterior_updates=300,new_preparation_episodes=2000,new_preparation_observations=60000))
 print('NATIVE_PREFLIGHT_PASS',flush=True)

def report():
 reference=read(M/'reports/moe_sboed_v5_history_canonical_20261119/RESULTS.json');vectors={};rows={};comparisons={}
 for key,r in reference['rows'].items():
  assert p.sha(r['checkpoint'])==r['checkpoint_sha256'];v=np.load(r['scores']);vectors['reference_'+key]=v
  rows['reference_'+key]=dict(method=('Previous V5 '+key if key in NAMES else r['method']),**p.engine.summaries(v))
 for arm in NAMES:
  path=O/arm/'RESULTS.json'
  if path.exists():
   r=read(path);assert r['evaluation']['bank_sha256']==p.engine.LOCK['bank_sha256'];v=np.load(O/arm/'evaluation/scores.npy');vectors[arm]=v;rows[arm]=dict(method=NAMES[arm],**p.engine.summaries(v))
 if 'learned' in vectors:
  for key in ['uniform','reference_learned','reference_previous_learned','reference_vsoed']:
   if key in vectors:comparisons['newV5_minus_'+key]=p.engine.summaries(vectors['learned']-vectors[key])
 if 'uniform' in vectors:comparisons['uniform_minus_previous_uniform']=p.engine.summaries(vectors['uniform']-vectors['reference_uniform'])
 status='COMPLETE' if all(k in vectors for k in NAMES) else 'PENDING'
 put(R/'RESULTS.json',dict(status=status,rows=rows,comparisons=comparisons,protocol=p.engine.LOCK,automatic_expansion=False,extra_neural_networks=0,extra_parameters=0))
 text='# Policy-only history-belief V5,seed101\n\nStatus:'+status+'. Native critic and posterior unchanged;original canonical evaluation.\n\n|Method|EIG lower bound|MCSE|\n|---|---:|---:|\n'
 for key in ['learned','uniform','reference_learned','reference_uniform','reference_previous_learned','reference_vsoed']:
  if key in rows:r=rows[key];text+=f"|{r['method']}|{r['mean']:.5f}|{r['MCSE']:.5f}|\n"
  else:text+=f'|{NAMES[key]}|PENDING|PENDING|\n'
 text+='\n|Paired comparison|Difference|95% case CI|\n|---|---:|---|\n'
 for key,r in comparisons.items():text+=f"|{key}|{r['mean']:+.5f}|[{r['CI95'][0]:+.5f},{r['CI95'][1]:+.5f}]|\n"
 text+='\nOne exploratory continuation seed,not across-seed robustness. Known finite-contrast saturation is not covered by MCSE. No automatic expansion. Distinct features/weights alone are not useful-specialization or routing evidence.\n';(R/'RESULTS.md').write_text(text)

def train(arm):
 bind();p.worker(arm,14400-cfg()['preparation_credit_seconds']);report()

def controller():
 lock=(O/'.pilot_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'LAUNCH.json').exists();put(O/'LAUNCH.json',dict(pid=os.getpid(),source=source_identity(),started=time.time()))
 jobs=['native_preflight','learned','uniform'];done=[]
 for name in jobs:
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
   put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB',next=name,completed=done,pending=jobs[len(done):]));time.sleep(20)
  assert source_identity()==read(O/'SOURCE_MANIFEST.json')
  put(O/'STATUS.json',dict(status='RUNNING',current=name,completed=done,pending=jobs[len(done)+1:]))
  task='smoke' if name=='native_preflight' else 'train';seconds=cfg()['native_preflight_max_seconds'] if task=='smoke' else int(14370-cfg()['preparation_credit_seconds'])
  command=['timeout','--signal=TERM','--kill-after=30',str(seconds),sys.executable,'-B','-u',str(M/'tools/moe_sboed_v5_history_belief_20261120.py'),'--task',task]
  if task=='train':command+=['--arm',name]
  with (O/(name+'.log')).open('x') as log:
   child=subprocess.Popen(command,cwd=M,stdout=log,stderr=subprocess.STDOUT);put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),worker_pid=child.pid,current=name,seed=101));code=child.wait()
  if code:put(O/'STATUS.json',dict(status='FAILED_PRESERVED',current=name,returncode=code,completed=done));report();return
  done.append(name);report()
 put(O/'STATUS.json',dict(status='COMPLETE',completed=done,automatic_expansion=False))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['prepare','smoke','train','report','controller'],required=True);parser.add_argument('--arm',choices=list(NAMES),default='learned');args=parser.parse_args()
 if args.task=='prepare':prepare()
 elif args.task=='smoke':smoke()
 elif args.task=='train':train(args.arm)
 elif args.task=='controller':controller()
 else:report()
if __name__=='__main__':main()
