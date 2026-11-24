"""Seed101 only: four matched V4 one-hot expert policies, native EIG unchanged."""
import argparse,copy,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
M=Path.home()/'Documents/objective_power_grid'
O=M/'experiments/location_finding/moe_sboed_v4_onehot_seed101_20261118'
R=M/'reports/moe_sboed_v4_onehot_seed101_20261118'
SELF=M/'tools/moe_sboed_v4_onehot_seed101_20261118.py'
spec=importlib.util.spec_from_file_location('onehot_frozen_replication',M/'tools/moe_sboed_v4_replication_20261117.py')
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
p=v.p;engine=p.engine
import torch,numpy as np

def read(path):return json.loads(Path(path).read_text())
def put(path,data):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(Path(path),data)

class OneHotActor(p.V4Actor):
    def configure(self,phase):
        assert phase in ['warmup','joint','frozen'];self.requires_grad_(False)
        if phase!='frozen':
            self.query.commander.requires_grad_(True);self.query.offset.requires_grad_(True)
            self.query.history.requires_grad_(phase=='joint')
            e=self.experts[self.active_expert];e.semantic.configure();e.projection.requires_grad_(True)
            self.action_head.requires_grad_(True);self.shared_small.requires_grad_(True)
        self.query.inference.eval()
        for e in self.experts:
            for name in ['anchor1','anchor3','opportunity']:
                if hasattr(e.semantic,name):getattr(e.semantic,name).eval()
        for parameter in self.parameters():parameter.grad=None
    def compute(self,x,collect=False):
        c=self.query(x);z,raw=self.experts[self.active_expert](c)
        # Exactly one-hot fusion: sum_k 1[k=active] z_k = z_active.
        action,info=self.decode(x,z[:,None],x.new_ones((len(x),1)))
        if collect:
            gates=x.new_zeros((len(x),4));gates[:,self.active_expert]=1
            return action,dict(info,z=z[:,None],raw=raw[:,None],gates=gates)
        return action

def build(k,device='cuda'):
    v.verify_frozen();state=p.load_parent(device);a=state['algorithm']
    actor=v.new_actor(a.actor_net,'learned',101)
    digest=p.n.digest(actor.state_dict());expected=read(v.O/'SEED_SMOKE.json')['initializations']['101']['actor_digest']
    assert digest==expected,'Initial weights must exactly match learned/uniform V4 seed101'
    actor.__class__=OneHotActor;actor.active_expert=k;actor.configure('joint')
    assert p.n.digest(actor.state_dict())==digest
    a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor);a.actor_target_net.configure('frozen')
    rate=a.actor_optimizer.param_groups[0]['lr']
    parameters=[q for q in actor.parameters() if q.requires_grad]
    a.actor_optimizer=torch.optim.Adam(parameters,lr=rate)
    a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999)
    ids={id(q) for group in a.actor_optimizer.param_groups for q in group['params']}
    assert not any(id(q) in ids for q in actor.router.parameters())
    for j,e in enumerate(actor.experts):
        if j!=k:assert not any(id(q) in ids for q in e.parameters())
    if device=='cuda':p.n.original.restore(state['rng'])
    engine.seed_all(101)
    return state,a,digest

def prepare():
    O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    assert not (R/'PROTOCOL.json').exists()
    cfg=dict(seed=101,seeds=[101],methods=[f'moe-sboed-e{k}' for k in range(1,5)],
      training_role='New trained one-hot baselines;not the preserved inherited inference ablations',
      initialization='Exact original V4 seed101 initial actor weights and frozen V2 runtime16001,not terminal learned V4 weights',
      native_objective='Original variational actor-critic full-horizon EIG;discount1,use_PCE=False;no auxiliary objectives',
      weights=[[int(i==j) for i in range(4)] for j in range(4)],
      actor_updates=2000,warmup_updates=200,joint_updates=1800,critic_updates=10000,posterior_updates=300000,
      new_episodes=2000000,T=30,new_observations=60000000,hard_total_seconds=14400,
      trainable='Selected specialist,commander/stage offsets,action head and weak correction;query history unfreezes at joint phase',
      frozen='Router,three unselected specialists,belief and anchor/opportunity providers',
      evaluation=engine.LOCK,checkpoint_selection='Fixed terminal2000 only;no test-bank model selection',
      no_other_seeds_authorized=True,source_sha256=p.sha(SELF),
      overhead='Four disposable native-update smoke tests are recorded separately and never used as final initialization')
    put(R/'PROTOCOL.json',cfg);print(json.dumps(cfg))

def smoke():
    p.setup();out=O/'smoke';assert not out.exists();out.mkdir();rows=[]
    with p.lease(),engine.Resources(out):
        for k in range(4):
            state,a,digest=build(k);actor=a.actor_net;folder=out/f'e{k+1}';folder.mkdir()
            data=np.load(p.n.BASE/'evaluation/diagnostic_candidate_histories.npz')
            x=torch.tensor(data['X'][:2,[0,14,29]].copy(),device='cuda').reshape(-1,117)
            calls=[];hooks=[]
            for j,e in enumerate(actor.experts):hooks.append(e.register_forward_hook(lambda *args,j=j:calls.append(j)))
            hooks.append(actor.router.register_forward_hook(lambda *args:calls.append('router')))
            with torch.no_grad():
                actual,info=actor(x,collect=True)
                assert calls==[k] and torch.equal(info['gates'],x.new_tensor([[int(i==k) for i in range(4)]]).expand(len(x),-1))
                z,_,_=p.V4Actor.packets(actor,x);expected,_=actor.decode(x,z,info['gates'])
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            for h in hooks:h.remove()
            target_before=p.n.digest(actor.state_dict());start=time.monotonic()
            receipt=engine.train_phase(a,folder,'warmup',1,start,start+300)
            assert receipt['counts']==dict(actor=1,critic=5,posterior=150)
            assert p.n.digest(actor.state_dict())!=target_before
            row=dict(expert=k+1,initial_actor_digest=digest,exact_onehot=True,only_selected_expert_executed=True,
              router_excluded_from_optimizer=True,unselected_experts_excluded_from_optimizer=True,
              native_update=receipt,gradient_norms=read_lines(folder/'TRAINING_METRICS.jsonl')[-1]['gradient_group_norms'])
            assert row['gradient_norms'][f'experts.{k}']>0
            rows.append(row);del actor,state,a;torch.cuda.empty_cache()
    put(R/'SMOKE.json',dict(status='PASS',seed=101,tests=rows,discarded=True,disposable_new_episodes=4000,disposable_observations=120000))
    print(json.dumps(dict(status='PASS',seconds_per_update=[r['native_update']['seconds'] for r in rows])),flush=True)

def read_lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]

def worker(k):
    start=time.monotonic();out=O/f'e{k+1}';assert not out.exists();out.mkdir();p.setup()
    assert p.sha(SELF)==read(R/'PROTOCOL.json')['source_sha256']
    with p.lease(),engine.Resources(out):
        actor=None
        try:
            state,a,digest=build(k);actor=a.actor_net
            put(out/'INITIALIZATION.json',dict(seed=101,expert=k+1,actor_digest=digest,
              parent_actor_sha256=p.n.ACTOR_SHA,parent_runtime_sha256=p.n.RUNTIME_SHA,parent_update=int(a.update),
              matched_V4_learned_uniform_initialization=True,weights=[int(i==k) for i in range(4)],training_rng_seed=101))
            torch.cuda.reset_peak_memory_stats();phases=[]
            for phase,steps in [('warmup',200),('joint',1800)]:phases.append(engine.train_phase(a,out,phase,steps,start,start+12600))
            training_seconds=time.monotonic()-start
            with (out/'runtime.pkl').open('wb') as f:p.n.cloudpickle.dump(dict(algorithm=a,rng=p.n.original.rng(),seed=101,expert=k+1),f,protocol=5)
            actor.configure('frozen');actor.eval();del state,a;torch.cuda.empty_cache()
            label=f'moe-sboed-e{k+1}';scores,evaluation=engine.evaluate(actor,out,label,start+14100,collect=False)
            assert sum(r['updates'] for r in phases)==2000
            assert [sum(r['counts'][key] for r in phases) for key in ['actor','critic','posterior']]==[2000,10000,300000]
            assert time.monotonic()-start<14400
            put(out/'RESULTS.json',dict(status='COMPLETE',method=label,seed=101,training_role='Matched trained onehot;not inherited inference ablation',
              phases=phases,training_seconds=training_seconds,total_seconds=time.monotonic()-start,
              new_episodes=2000000,new_observations=60000000,weights=[int(i==k) for i in range(4)],
              evaluation=evaluation,checkpoint=str(out/'joint_actor.pt'),checkpoint_sha256=p.sha(out/'joint_actor.pt')))
            put(out/'STATUS.json',dict(status='COMPLETE',method=label,seed=101))
        except Exception as error:
            if actor is not None:torch.save(dict(actor_state=actor.state_dict(),incomplete=True,error=repr(error)),out/'incomplete_actor.pt')
            put(out/'STATUS.json',dict(status='FAILED_OR_CAPPED',error=repr(error),seconds=time.monotonic()-start));raise

def report():
    rows=[]
    for k in range(4):
        path=O/f'e{k+1}/RESULTS.json'
        if path.exists():rows.append(read(path))
    put(R/'RESULTS.json',dict(status='COMPLETE' if len(rows)==4 else 'PENDING',required_seed=101,completed=len(rows),required=4,rows=rows))
    text='# Trained V4 one-hot experts: seed101 only\n\nThese are newly trained one-hot policies with matched initial V4 weights and native2,000-update training. Inherited one-hot ablations remain separately preserved. No seeds102–105 are scheduled.\n\n|Method|Seed101 EIG|MCSE|Status|\n|---|---:|---:|---|\n'
    for k in range(4):
        d=next((r for r in rows if r['method']==f'moe-sboed-e{k+1}'),None)
        text+=f"|moe-sboed-e{k+1}|"+(f"{d['evaluation']['mean']:.5f}|{d['evaluation']['MCSE']:.5f}|COMPLETE|\n" if d else 'PENDING|PENDING|PENDING|\n')
    (R/'RESULTS.md').write_text(text)

def controller():
    lock=(O/'.controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(R/'SMOKE.json')['status']=='PASS';done=[];failed=[]
    for k in range(4):
        out=O/f'e{k+1}'
        if (out/'RESULTS.json').exists() and read(out/'RESULTS.json')['status']=='COMPLETE':done.append(k+1);continue
        assert not out.exists(),'Partial job preserved;no silent retry'
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU',seed=101,expert=k+1));time.sleep(20)
        put(O/'STATUS.json',dict(status='RUNNING',seed=101,expert=k+1,completed=done,pending=list(range(k+2,5)),failed=failed))
        with (O/f'e{k+1}.log').open('x') as f:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30','14370',sys.executable,'-B','-u',str(SELF),'--task','worker','--expert',str(k+1)],stdout=f,stderr=subprocess.STDOUT,cwd=M)
            put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),pid=child.pid,seed=101,expert=k+1));code=child.wait()
        if code:failed.append(dict(expert=k+1,returncode=code))
        else:done.append(k+1)
        report()
    put(O/'STATUS.json',dict(status='COMPLETE' if not failed else 'FINISHED_WITH_FAILURES',seed=101,completed=done,failed=failed))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['prepare','smoke','controller','worker','report'],required=True);parser.add_argument('--expert',type=int,choices=[1,2,3,4]);a=parser.parse_args()
    if a.task=='prepare':prepare()
    elif a.task=='smoke':smoke()
    elif a.task=='worker':worker(a.expert-1)
    elif a.task=='controller':controller()
    else:report()
if __name__=='__main__':main()
