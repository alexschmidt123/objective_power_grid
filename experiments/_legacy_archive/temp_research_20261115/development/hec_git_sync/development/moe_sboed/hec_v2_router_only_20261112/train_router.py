"""Matched native V2 continuation: change only actor requires_grad selection."""
import argparse, gc, hashlib, io, json, os, pathlib, signal, subprocess, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = pathlib.Path.home() / 'Documents/objective_power_grid_temp'
WORK = ROOT / 'development/hec_git_sync'
BASE = ROOT / 'scratch_runs/moe_hec_v2_icml_20261109'
os.environ['MOE_V2_RUN_ROOT'] = str(BASE)
sys.path.insert(0, str(BASE / 'source'))
from native_common import original, cloudpickle, allopts, digest, P
import restore_v1, evaluate_v2
sys.path.insert(0, str(WORK / 'development/moe_sboed/hec_v2_resolution_20261112'))
from run_resolution import load_library
from resolution import nested_scores, summary, paired

ACTOR_SHA = '90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108'
RUNTIME_SHA = 'd38c09167c4936a6a0660d16079ecc3d5e1108c32bdbc840cc8851c1ef7266b1'
PARENT_UPDATE = 16001
STOP = False

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''): h.update(block)
    return h.hexdigest()

def put(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def load_parent():
    assert sha(BASE/'arms/candidate/actor.pt') == ACTOR_SHA
    assert sha(BASE/'arms/candidate/runtime.pkl') == RUNTIME_SHA
    old = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location=lambda s,l: s.cuda(0) if l.startswith('cuda') else s, weights_only=False)
    try:
        with (BASE/'arms/candidate/runtime.pkl').open('rb') as f: state = cloudpickle.load(f)
    finally: torch.storage._load_from_bytes = old
    a = state['algorithm']
    assert int(a.update) == PARENT_UPDATE and a.discount == 1 and not a.use_PCE
    assert a.n_batch == 10000 and a.n_newtraj == 1000
    assert len(a._runtime_buffer['params']) == 1000000
    cp = torch.load(BASE/'arms/candidate/actor.pt', map_location='cuda', weights_only=True)['actor_state']
    assert all(torch.equal(v, a.actor_net.state_dict()[k]) for k,v in cp.items())
    for opt in allopts(a):
        opt._optimizer_step_pre_hooks.clear(); opt._optimizer_step_post_hooks.clear()
    if hasattr(a, '_soft_tasks'): a._soft_tasks.enabled = False
    return state

def configure(actor, arm):
    before = digest(actor.state_dict())
    if arm == 'router_only':
        for name,p in actor.named_parameters():
            p.requires_grad_(name.startswith(('coordinator.', 'response_gate.')))
            p.grad = None
    else: assert arm == 'ordinary'
    assert before == digest(actor.state_dict())
    return [name for name,p in actor.named_parameters() if p.requires_grad]

def frozen_digest(actor):
    return digest({k:p for k,p in actor.named_parameters() if not p.requires_grad})

class Halt(Exception): pass
class Table:
    def __init__(self): self.data = {}
    def record(self,k,v): self.data[k] = v
    def clear(self): self.data.clear()

class Logger:
    def __init__(self,a,arm,out,cfg,start):
        self.algorithm=a; self.arm=arm; self.out=out; self.cfg=cfg
        self.start=time.monotonic(); self.global_start=start
        self.logger=self; self.tabular=Table()
        self.counts=dict(actor=0,critic=0,posterior=0); self.gradients={}
    def push_prefix(self,*args): pass
    def pop_prefix(self,*args): pass
    def log(self,*args): pass
    def dump_all(self):
        a=self.algorithm; n=int(a.update)-PARENT_UPDATE
        assert self.counts == dict(actor=n,critic=5*n,posterior=150*n)
        if n==1 or n%25==0 or n==self.cfg['updates']:
            row=dict(status='TRAINING',arm=self.arm,updates=n,target=self.cfg['updates'],
                     seconds=time.monotonic()-self.start,counts=self.counts.copy(),
                     new_episodes=1000*n,new_observations=30000*n,
                     actor_lr=a.actor_optimizer.param_groups[0]['lr'],
                     gradient_norms=self.gradients,gpu_peak_bytes=torch.cuda.max_memory_allocated())
            put(self.out/'STATUS.json',row)
            with (self.out/f'{self.arm}_metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if STOP or (self.out/'STOP_REQUESTED').exists() or time.monotonic()-self.global_start>self.cfg['wall_seconds']:
            raise Halt('Bounded stopping rule; incomplete artifacts retained')

def validate(a, out):
    data=np.load(BASE/'evaluation/diagnostic_candidate_histories.npz')
    x=torch.tensor(data['X'][:8,[0,14,29]].copy(),device='cuda').reshape(-1,117)
    with torch.no_grad(): before=a.actor_net(x).clone()
    names=configure(a.actor_net,'router_only')
    with torch.no_grad(): after=a.actor_net(x)
    assert torch.equal(before,after)
    trainable=[p for p in a.actor_net.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable)==25924
    opt_ids={id(p) for g in a.actor_optimizer.param_groups for p in g['params']}
    assert all(id(p) in opt_ids and p in a.actor_optimizer.state for p in trainable)
    loss=-a.critic_net(torch.cat([x,a.actor_net(x)],-1)).mean()
    loss.backward()
    norms={}
    for name,p in a.actor_net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
            key=name.split('.')[0]; norms[key]=norms.get(key,0)+float(p.grad.square().sum())
        else: assert p.grad is None,name
    assert all(v>0 for v in norms.values())
    with torch.no_grad():
        uniform=a.actor_net(x,routing_override=x.new_full((4,),.25))
        sensitivity=float((uniform-before).abs().max())
    assert sensitivity>0
    for p in a.actor_net.parameters(): p.grad=None
    result=dict(status='PASS',exact_forward_equivalence=True,trainable_parameters=25924,
                trainable_names=names,native_actor_loss_gradient_norms={k:v**.5 for k,v in norms.items()},
                frozen_parameter_gradients_absent=True,optimizer_state_membership_preserved=True,
                learned_vs_uniform_max_action_difference=sensitivity,optimizer_steps=0)
    put(out/'VALIDATION.json',result); print(json.dumps(result),flush=True)

def train_arm(arm,out,cfg,start):
    state=load_parent(); a=state['algorithm']; names=configure(a.actor_net,arm)
    root=out/arm; root.mkdir()
    frozen_before=frozen_digest(a.actor_net); buffers_before=digest(dict(a.actor_net.named_buffers()))
    initial=digest(a.actor_net.state_dict()); router_before=digest({k:v for k,v in a.actor_net.state_dict().items() if k.startswith(('coordinator.','response_gate.'))})
    l=Logger(a,arm,out,cfg,start); a.dowel=l; a.post_approx.dowel=l; a.save_path=None
    for op,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(o,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for o in oo]:
        def count(*args,key=key,**kwargs): l.counts[key]+=1
        op.register_step_post_hook(count)
    def monitor(*args):
        if l.counts['actor']%25: return
        sums={}
        for name,p in a.actor_net.named_parameters():
            if not p.requires_grad: assert p.grad is None,name
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(),name
                key=name.split('.')[0]; sums[key]=sums.get(key,0)+float(p.grad.square().sum())
        l.gradients={k:v**.5 for k,v in sums.items()}
        if arm=='router_only': assert all(l.gradients.get(k,0)>0 for k in ['coordinator','response_gate'])
    a.actor_optimizer.register_step_pre_hook(monitor)
    original.restore(state['rng']); torch.cuda.reset_peak_memory_stats()
    put(root/'STARTED.json',dict(pid=os.getpid(),parent_update=PARENT_UPDATE,
         initial_actor_digest=initial,trainable_names=names,
         trainable_parameters=sum(p.numel() for p in a.actor_net.parameters() if p.requires_grad),
         optimizer_and_rng_inherited=True))
    profile=json.loads((P/'frozen_protocol.json').read_text())['profile']; profile['discount']=1.
    status='COMPLETE'
    try: original.train_native(a,l,profile,target=PARENT_UPDATE+cfg['updates'])
    except Halt: status='INCOMPLETE'
    n=int(a.update)-PARENT_UPDATE
    assert frozen_before==frozen_digest(a.actor_net)
    assert buffers_before==digest(dict(a.actor_net.named_buffers()))
    router_after=digest({k:v for k,v in a.actor_net.state_dict().items() if k.startswith(('coordinator.','response_gate.'))})
    if status=='COMPLETE': assert n==cfg['updates'] and router_after!=router_before
    for op in allopts(a): op._optimizer_step_pre_hooks.clear(); op._optimizer_step_post_hooks.clear()
    torch.save(dict(actor_state=a.actor_net.state_dict(),native_iteration=int(a.update)),root/'actor.pt')
    with (root/'runtime.pkl').open('wb') as f:
        cloudpickle.dump(dict(algorithm=a,rng=original.rng(),method='native_v2_'+arm),f,protocol=5)
    result=dict(status=status,arm=arm,updates=n,counts=l.counts,seconds=time.monotonic()-l.start,
                actor_sha256=sha(root/'actor.pt'),runtime_sha256=sha(root/'runtime.pkl'),
                frozen_actor_parameters_preserved=True,actor_buffers_preserved=True,
                router_changed=router_after!=router_before,new_episodes=n*1000,new_observations=n*30000)
    put(root/'RESULT.json',result); print('ARM_END',json.dumps(result),flush=True)
    return status

def evaluate(out,cfg,start):
    lib=load_library(); actor=lib.load_frozen('cuda'); rng=np.random.default_rng(cfg['case_seed'])
    n=cfg['evaluation_cases']; theta=torch.tensor(rng.standard_normal((n,4)),device='cuda')
    noise=torch.tensor(rng.standard_normal((n,30)),device='cuda')
    np.savez_compressed(out/'FRESH_CASES.npz',truth=theta.cpu().numpy(),noise=noise.cpu().numpy())
    bank=np.random.default_rng(cfg['contrast_seed']).standard_normal((cfg['contrasts'],4))
    put(out/'EVALUATION_BANK.json',dict(seed=cfg['contrast_seed'],sha256=hashlib.sha256(bank.view(np.uint8)).hexdigest()))
    contrasts=torch.tensor(bank,device='cuda'); del bank
    scores={}; methods={}; deadline=start+cfg['wall_seconds']
    for mode in ['original','original_uniform','ordinary','router_only','router_only_uniform']:
        if mode in ['ordinary','router_only']:
            actor.load_state_dict(torch.load(out/mode/'actor.pt',map_location='cuda',weights_only=True)['actor_state'])
        vals=[]; tick=time.monotonic()
        for first in range(0,n,16):
            th,eps=theta[first:first+16],noise[first:first+16]
            actions,obs,_=lib.rollout(actor,th,eps,'uniform' if mode.endswith('_uniform') else 'original',deadline=deadline)
            v=nested_scores(actions,obs,th,contrasts,[cfg['contrasts']],deadline=deadline)[cfg['contrasts']]
            vals.append(v.cpu().numpy())
            put(out/'STATUS.json',dict(status='EVALUATING',policy=mode,completed=min(first+16,n),total=n,seconds=time.monotonic()-start))
        scores[mode]=np.concatenate(vals); np.save(out/f'{mode}_scores.npy',scores[mode])
        methods[mode]=summary(scores[mode],cfg['contrasts']); methods[mode]['seconds']=time.monotonic()-tick
    result=dict(status='COMPLETE',primary_router_only_minus_ordinary=paired(scores['router_only'],scores['ordinary']),
                secondary_router_only_minus_original=paired(scores['router_only'],scores['original']),
                secondary_router_only_minus_own_uniform=paired(scores['router_only'],scores['router_only_uniform']),
                secondary_router_only_minus_original_uniform=paired(scores['router_only'],scores['original_uniform']),
                methods=methods,seconds=time.monotonic()-start,
                limitations='One inherited training lineage; 16M prior-sPCE is a finite-contrast lower bound. Secondary intervals unadjusted.')
    put(out/'RESULTS.json',result); print('FINAL',json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--validate-only',action='store_true'); p.add_argument('--run',action='store_true'); args=p.parse_args()
    if args.run==args.validate_only: p.error('Choose --run or --validate-only')
    out=args.output.expanduser().resolve()
    assert ROOT/'scratch_runs' in out.parents and not out.exists()
    out.mkdir(parents=True)
    cfg=dict(updates=2000,evaluation_cases=2048,contrasts=16000000,case_seed=202611129101,
             contrast_seed=202611129102,wall_seconds=14400,arms=['router_only','ordinary'],
             primary='router_only minus equal-update ordinary V2 continuation; paired two-sided 95% t interval',
             checkpoint_selection='Final update only; no selection using evaluation outcomes',
             native_objective_unchanged=True,source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    put(out/'CONFIG.json',cfg)
    protected=[BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl',*sorted((BASE/'source').glob('*.py')),pathlib.Path(__file__).resolve()]
    manifest={str(f):sha(f) for f in protected}; put(out/'PRE_RUN_MANIFEST.json',manifest)
    original.setup()
    while True:
        try: lease=original.acquire(); break
        except BlockingIOError:
            put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU',pid=os.getpid())); time.sleep(30)
    start=time.monotonic()
    def stop(*args):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop)
    try:
        state=load_parent(); validate(state['algorithm'],out); del state; gc.collect(); torch.cuda.empty_cache()
        if args.validate_only:
            put(out/'STATUS.json',dict(status='VALIDATED',optimizer_steps=0)); return
        for arm in cfg['arms']:
            status=train_arm(arm,out,cfg,start); gc.collect(); torch.cuda.empty_cache()
            if status!='COMPLETE':
                put(out/'STATUS.json',dict(status=status,arm=arm)); return
        evaluate(out,cfg,start)
        assert all(sha(path)==value for path,value in manifest.items())
        put(out/'STATUS.json',dict(status='COMPLETE',seconds=time.monotonic()-start,historical_artifacts_preserved=True))
    except Exception as exc:
        put(out/'STATUS.json',dict(status='FAILED',type=type(exc).__name__,message=str(exc))); raise
    finally:
        for f in lease: f.close()

if __name__=='__main__': main()
