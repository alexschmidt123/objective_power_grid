"""Exploratory five-seed V4 continuation replication;frozen pilot architecture.

Only seeds,metadata and output/evaluation locations are parameterized in memory.
Original pilot source,configs,banks,models and results are never rewritten.
"""
import argparse,copy,fcntl,hashlib,importlib.util,json,os,pathlib,signal,subprocess,sys,time
M=pathlib.Path.home()/'Documents/objective_power_grid'
P=M/'experiments/location_finding/moe_sboed_v4_20261116'
O=M/'experiments/location_finding/moe_sboed_v4_replication_20261117'
R=M/'reports/moe_sboed_v4_replication_20261117'
C=M/'configs/moe_sboed_v4_replication_20261117.json'
SELF=M/'tools/moe_sboed_v4_replication_20261117.py'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('frozen_v4_pilot',M/'src/research/moe_sboed_v4_20261116/run.py')
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
import numpy as np
import torch
from scipy.stats import t as student
STUDY='EXPLORATORY_REPLICATION_AFTER_FAILED_PILOT_EXPANSION_CRITERION'
NAMES={'learned':'MoE-sBOED-V4','uniform':'MoE-sBOED-V4-uniform'}
STOP=False


def read(path):return json.loads(path.read_text())
def cfg():return read(C)
def put(path,value):p.put(path,value)
def sha(path):return p.sha(path)


def verify_frozen():
    c=cfg()
    for name,value in c['frozen_source_sha256'].items():assert sha(M/name)==value,name
    for arm in NAMES:
        assert sha(P/arm/'joint_actor.pt')==c['seed101_checkpoints'][arm]['sha256']
    assert sha(M/'configs/moe_sboed_v4_20261116.json')==c['original_protocol_sha256']


def resolved(seed):return read(M/f'configs/moe_sboed_v4_replication_20261117_seed{seed}.json')


def source_identity(seed):
    paths=list(cfg()['frozen_source_sha256'])+[str(SELF.relative_to(M)),str(C.relative_to(M)),
          f'configs/moe_sboed_v4_replication_20261117_seed{seed}.json']
    return {name:sha(M/name) for name in paths}


def new_actor(parent,arm,seed):
    # Exact pilot recipe when seed=101;all fresh template innovations vary across seeds too.
    template=p.StudyMoE(parent,202611150000+seed,'uniform')
    p.engine.seed_all((202611164000+seed) % (2**32))
    return p.V4Actor(template,arm)


def build(arm,device,seed):
    state=p.load_parent(device);a=state['algorithm'];actor=new_actor(a.actor_net,arm,seed)
    a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor);a.actor_target_net.configure('frozen')
    p.engine.optimizer(a)
    if device=='cuda':p.n.original.restore(state['rng'])
    p.engine.seed_all(seed)
    # Fingerprints are captured before any simulator/replay sampling.
    a._v4_replication_initial_rng=dict(cpu=hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        cuda=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in torch.cuda.get_rng_state_all()] if device=='cuda' else [])
    return state,a


def bank():
    path=O/'evaluation_bank.npz';identity=read(O/'BANK_IDENTITY.json')
    assert sha(path)==identity['sha256']
    b=np.load(path);assert b['truth'].shape==(2000,4) and b['noise'].shape==(2000,30) and b['contrasts'].shape==(1000000,4)
    return b


def bank_alias(folder):
    folder.mkdir(exist_ok=True,parents=True)
    link=folder/'development_validation_bank.npz'
    if link.exists():assert link.resolve()==(O/'evaluation_bank.npz').resolve()
    else:link.symlink_to(O/'evaluation_bank.npz')


def configure(seed,arm,training=False):
    """Adapt the frozen runner's seed/path interfaces,not its training implementation."""
    verify_frozen();local=resolved(seed)
    p.C=M/f'configs/moe_sboed_v4_replication_20261117_seed{seed}.json'
    p.cfg=lambda:local
    p.O=O/f'seed{seed}'
    bank_alias(p.O)
    p.bank=bank
    p.new_actor=lambda parent,kind:new_actor(parent,kind,seed)
    p.build=lambda kind,device:build(kind,device,seed)
    p.source_identity=lambda:source_identity(seed)
    original_put=p.put
    def metadata(value):
        if isinstance(value,dict):
            return {k:(seed if k=='seed' and v==101 else metadata(v)) for k,v in value.items()}
        if isinstance(value,list):return [metadata(v) for v in value]
        return value
    def named_put(path,value):
        value=metadata(value)
        if isinstance(value,dict):
            value.update(seed=seed,study_type=STUDY)
            if pathlib.Path(path).name=='INITIALIZATION.json':
                value.update(training_seed=seed,template_initialization_seed=202611150000+seed,
                    v4_initialization_seed=202611164000+seed)
                expected=read(O/'SEED_SMOKE.json')['initializations'][str(seed)]['actor_digest']
                assert value['actor_digest']==expected,'CPU/GPU initialization mismatch'
                (pathlib.Path(path).parent/'rng_identity.json').write_text(json.dumps(rng_identity(),indent=2)+'\n')
        original_put(path,value)
    p.put=named_put;p.engine.put=named_put
    if training:
        p.O.mkdir(exist_ok=True)
        manifest=p.O/'SOURCE_MANIFEST.json'
        if manifest.exists():assert read(manifest)==source_identity(seed)
        else:manifest.write_text(json.dumps(source_identity(seed),indent=2)+'\n')
        original_dump=p.n.cloudpickle.dump
        def dump(value,file,*args,**kwargs):
            if isinstance(value,dict) and 'algorithm' in value:
                value=dict(value,seed=seed,training_seed=seed,study_type=STUDY,
                           initialization_seed=202611164000+seed,template_initialization_seed=202611150000+seed)
            return original_dump(value,file,*args,**kwargs)
        p.n.cloudpickle.dump=dump


def rng_identity():
    return dict(training_seed=CURRENT_SEED,
        cpu=hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        cuda=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def prepare():
    verify_frozen();O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    path=O/'evaluation_bank.npz';assert not path.exists(),'Never regenerate or search evaluation banks'
    rng=np.random.default_rng(cfg()['evaluation_seed'])
    np.savez(path,truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)))
    put(O/'BANK_IDENTITY.json',dict(status='LOCKED_BEFORE_ANY_REPLICATION_EVALUATION',sha256=sha(path),
        seed=cfg()['evaluation_seed'],cases=2000,contrasts=1000000,T=30,study_type=STUDY,
        scorer_sha256=cfg()['scorer_sha256'],prior='N(0,I4)',sigma=.5,bounds=[[-4,4],[-4,4]]))
    print(json.dumps(dict(bank=str(path),sha256=sha(path))))


@torch.no_grad()
def smoke():
    verify_frozen();torch.set_num_threads(2);state=p.load_parent('cpu');parent=state['algorithm'].actor_net
    initializations={};rngs={}
    original101=[read(P/arm/'INITIALIZATION.json')['actor_digest'] for arm in NAMES]
    for seed in range(101,106):
        aa=new_actor(parent,'learned',seed);bb=new_actor(parent,'uniform',seed)
        digest=p.n.digest(aa.state_dict());assert digest==p.n.digest(bb.state_dict())
        if seed==101:assert digest==original101[0]==original101[1],'Do not change original initialization recipe'
        components={}
        for name,module in [('router',aa.router),('head',aa.action_head),('weak',aa.shared_small),
                            *[(f'projection{k+1}',e.projection) for k,e in enumerate(aa.experts)],
                            *[(f'innovation{k+1}',e.semantic.innovation) for k,e in enumerate(aa.experts)]]:
            components[name]=p.n.digest(module.state_dict())
        x=torch.zeros(2,117);x[:,0]=1
        a,ai=aa(x,collect=True);b,bi=bb(x,collect=True)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        assert torch.equal(bi['gates'],torch.full_like(bi['gates'],.25))
        assert ai['z'].shape==(2,4,128) and torch.isfinite(a).all() and (a.abs()<=4).all()
        p.engine.seed_all(seed)
        rngs[str(seed)]=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        initializations[str(seed)]=dict(actor_digest=digest,template_seed=202611150000+seed,
             initialization_seed=202611164000+seed,new_component_digests=components,
             paired_common_values_bitwise_equal=True)
        del aa,bb
    assert len({v['actor_digest'] for v in initializations.values()})==5
    assert len(set(rngs.values()))==5
    for key in initializations['101']['new_component_digests']:
        assert len({v['new_component_digests'][key] for v in initializations.values()})==5,key
    put(O/'SEED_SMOKE.json',dict(status='PASS',device='CPU',original_seed101_reproduced=True,
        architecture_source_unchanged=True,initializations=initializations,training_rng_fingerprints=rngs,
        each_seed_new_components_distinct=True,original_test_bank_read=False))
    print(json.dumps(dict(status='PASS',seeds=list(initializations),original_seed101_reproduced=True)))


def train(seed,arm):
    assert seed in [102,103,104,105] and arm in NAMES
    configure(seed,arm,training=True)
    print(json.dumps(dict(study_type=STUDY,seed=seed,arm=arm,training_rng_seed=seed,
        initialization_seed=202611164000+seed,template_seed=202611150000+seed)),flush=True)
    p.worker(arm,14400)
    result=read(p.O/arm/'RESULTS.json');assert result['seed']==seed and result['updates']==2000
    assert result['new_training_episodes']==2000000
    assert sum(v['counts']['actor'] for v in result['phases'])==2000
    assert sum(v['counts']['critic'] for v in result['phases'])==10000
    assert sum(v['counts']['posterior'] for v in result['phases'])==300000
    if arm=='uniform':
        gates=np.load(p.O/arm/'evaluation/diagnostics.npz')['gates'];assert np.all(gates==.25)
        assert not any(k.startswith('router.') for k in result['training_gradient_norms'])
        state=torch.load(p.O/arm/'joint_actor.pt',map_location='cpu',weights_only=True)['actor_state']
        rd=p.n.digest({k[len('router.'):]:v for k,v in state.items() if k.startswith('router.')})
        assert rd==read(O/'SEED_SMOKE.json')['initializations'][str(seed)]['new_component_digests']['router']
    verify_frozen()
    p.put(p.O/arm/'REPLICATION_VALIDATION.json',dict(status='PASS',seed=seed,arm=arm,counts_verified=True,architecture_unchanged=True,uniform_verified=arm=='uniform'))


def eval101(arm):
    configure(101,arm);out=O/'seed101_evaluation'/arm;out.mkdir(parents=True,exist_ok=False)
    p.setup()
    with p.lease(),p.engine.Resources(out):
        state=p.load_parent('cuda');actor=new_actor(state['algorithm'].actor_net,arm,101).cuda()
        checkpoint=P/arm/'joint_actor.pt';before=sha(checkpoint)
        actor.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['actor_state'])
        actor.configure('frozen');actor.eval()
        result=p.evaluate(actor,out/'evaluation',NAMES[arm],time.monotonic()+1700)
        assert before==sha(checkpoint)
        p.put(out/'RESULTS.json',dict(status='COMPLETE',seed=101,evaluation=result,checkpoint=str(checkpoint),checkpoint_sha256=before,training_updates=0))
    verify_frozen()


def secondary(arm):
    configure(101,arm);p.O=O/'secondary_seed101';p.O.mkdir(exist_ok=True)
    bank_alias(p.O)
    p.previous(arm,1800);verify_frozen()


def training_path(seed,arm):return P/arm if seed==101 else O/f'seed{seed}'/arm
def evaluation_path(seed,arm):return O/'seed101_evaluation'/arm if seed==101 else training_path(seed,arm)


def report():
    import csv
    R.mkdir(parents=True,exist_ok=True);identity=read(O/'BANK_IDENTITY.json');records=[];scores={};inventory=[];costs=[];diagnostics={}
    for seed in range(101,106):
        for arm,name in NAMES.items():
            tr=training_path(seed,arm);ev=evaluation_path(seed,arm);result=ev/'RESULTS.json'
            cp=tr/'joint_actor.pt'
            if result.exists() and (seed==101 or (tr/'REPLICATION_VALIDATION.json').exists()):
                d=read(result);assert d['status']=='COMPLETE' and d['seed']==seed
                e=d['evaluation'];assert e['bank_sha256']==identity['sha256']
                v=np.load(ev/'evaluation/scores.npy');assert len(v)==2000 and np.isfinite(v).all()
                scores[seed,arm]=v;records.append(dict(seed=seed,method=name,EIG=float(v.mean()),MCSE=float(v.std(ddof=1)/2000**.5)))
                diagnostics[f'{name}/seed{seed}']=e.get('diagnostics')
            if (tr/'RESULTS.json').exists():
                td=read(tr/'RESULTS.json');costs.append(dict(seed=seed,method=name,training_seconds=td['training_seconds'],total_run_seconds=td['total_seconds'],episodes=td['new_training_episodes'],observations=td['new_training_observations'],phases=td['phases'],resources=read(tr/'RESOURCES.json') if (tr/'RESOURCES.json').exists() else None,evaluation_resources=read(ev/'RESOURCES.json') if (ev/'RESOURCES.json').exists() else None))
                diagnostics.setdefault(f'{name}/seed{seed}',{})
                if diagnostics[f'{name}/seed{seed}'] is not None:diagnostics[f'{name}/seed{seed}']['training_gradient_norms']=td['training_gradient_norms']
            init=read(tr/'INITIALIZATION.json') if (tr/'INITIALIZATION.json').exists() else None
            inventory.append(dict(seed=seed,method=name,checkpoint=str(cp),exists=cp.exists(),initialization=init,
                checkpoint_sha256=sha(cp) if cp.exists() else None,
                configuration=str(tr/'CONFIG.json'),initialization_path=str(tr/'INITIALIZATION.json'),
                training_log=str(tr/'TRAINING_METRICS.jsonl'),runtime=str(tr/'runtime.pkl'),evaluation=str(ev/'evaluation/RESULTS.json')))
    per_seed=[]
    for seed in range(101,106):
        if (seed,'learned') in scores and (seed,'uniform') in scores:
            ia,ib=[read(training_path(seed,a)/'INITIALIZATION.json') for a in NAMES]
            assert ia['actor_digest']==ib['actor_digest']
            per_seed.append(dict(seed=seed,**p.engine.summaries(scores[seed,'learned']-scores[seed,'uniform'])))
    summaries={}
    for arm,name in NAMES.items():
        vals=[scores[s,arm].mean() for s in range(101,106) if (s,arm) in scores]
        if len(vals)==5:
            per_case=np.stack([scores[s,arm] for s in range(101,106)]).mean(0)
            summaries[name]=dict(n_training_seeds=5,mean=float(np.mean(vals)),seed_SD=float(np.std(vals,ddof=1)),conditional_bank_MCSE_of_seed_average=float(per_case.std(ddof=1)/2000**.5))
    primary=None
    if len(per_seed)==5:
        delta=np.array([r['mean'] for r in per_seed]);sd=float(delta.std(ddof=1));se=sd/5**.5;q=float(student.ppf(.975,4))
        primary=dict(n_training_seeds=5,mean_delta=float(delta.mean()),SD_delta=sd,seed_SE=se,
                     CI95=[float(delta.mean()-q*se),float(delta.mean()+q*se)],positive_seeds=int((delta>0).sum()),
                     interval='Paired training-seed Student t,df4;conditional on one fixed bank')
    secondary_rows=[];secondary_differences={}
    for arm,name in [('previous_learned','Previous collaborative MoE-sBOED'),('previous_uniform','Previous collaborative MoE-sBOED-uniform')]:
        result=O/'secondary_seed101'/arm/'RESULTS.json'
        if result.exists():
            d=read(result);assert d['evaluation']['bank_sha256']==identity['sha256'];v=np.load(result.parent/'evaluation/scores.npy')
            secondary_rows.append(dict(method=name,training_seed_coverage=[101],EIG=float(v.mean()),MCSE=float(v.std(ddof=1)/2000**.5)))
            if (101,'learned') in scores:secondary_differences[name]=p.engine.summaries(scores[101,'learned']-v)
    complete=len(records)==10
    result=dict(status='COMPLETE' if complete else 'PENDING',study_type=STUDY,completed_V4_evaluations=len(records),required_V4_evaluations=10,
        records=records,per_seed_paired_differences=per_seed,method_summaries=summaries,primary_seed_level=primary,
        secondary_seed101_only=secondary_rows,secondary_case_differences_seed101_only=secondary_differences,
        bank_identity=identity,interpretation='Pending all five paired training seeds' if primary is None else
        ('Positive exploratory routing effect;not confirmatory and not proof of superiority over V3' if primary['CI95'][0]>0 else 'No reliable five-seed routing advantage established'),
        limitations=['Shared frozen V2 pretraining:independent continuation initialization/RNG only,not end-to-end independence.',
        'Expansion was explicitly authorized after original pilot failed its preregistered expansion criterion;all conclusions exploratory.',
        'MCSE conditions on the shared contrast bank;finite-contrast bias/contrast-bank uncertainty not covered.',
        'Same cases across seeds;10000 policy trajectories are not10000 training seeds. Seed-level inference uses n5.',
        'V3 secondary coverage is seed101 only on this bank;no five-seed V4-versus-V3 superiority claim.'])
    put(R/'RESULTS.json',result);put(R/'CHECKPOINT_MANIFEST.json',inventory);put(R/'COMPUTE.json',costs);put(R/'DIAGNOSTICS.json',diagnostics)
    with (R/'FIVE_SEED_RESULTS.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['Method','101','102','103','104','105','Mean','Training-seed SD'])
        for arm,name in NAMES.items():
            s=summaries.get(name,{});w.writerow([name,*[float(scores[k,arm].mean()) if (k,arm) in scores else 'PENDING' for k in range(101,106)],s.get('mean','PENDING'),s.get('seed_SD','PENDING')])
    text='# Exploratory five-seed V4 continuation replication\n\nThe seed101 pilot failed its predeclared expansion rule. Additional seeds were explicitly authorized after observing the routing effect. This is exploratory,not confirmatory. Original pilot results remain unchanged on their separate bank.\n\n'
    text+='| Method |101|102|103|104|105|Mean ± training-seed SD|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for arm,name in NAMES.items():
        vals=[f'{scores[k,arm].mean():.5f}' if (k,arm) in scores else 'PENDING' for k in range(101,106)];s=summaries.get(name)
        text+='|'+name+'|'+'|'.join(vals)+'|'+(f"{s['mean']:.5f} ± {s['seed_SD']:.5f}" if s else 'PENDING')+'|\n'
    text+='\n| Seed | Paired learned-minus-uniform | Evaluation MCSE | Within-seed95% CI |\n|---|---:|---:|---|\n'
    for v in per_seed:text+=f"|{v['seed']}|{v['mean']:+.5f}|{v['MCSE']:.5f}|[{v['CI95'][0]:+.5f},{v['CI95'][1]:+.5f}]|\n"
    if primary:text+='\nFive-seed paired result:\n\n```json\n'+json.dumps(primary,indent=2)+'\n```\n\n'+result['interpretation']+'\n'
    text+='\nSecondary comparison has previous collaborative checkpoints for seed101 ONLY on the same new bank:\n\n```json\n'+json.dumps(dict(rows=secondary_rows,paired_case_differences=secondary_differences),indent=2)+'\n```\n\n'
    text+='\n'.join('- '+x for x in result['limitations'])+'\n\nICML reporting:retain original pilot and this exploratory replication separately;report seed-level uncertainty and full compute/initialization lineage;do not equate a V4 routing gain with superior overall EIG. No further architectures or searches are scheduled.\n'
    (R/'RESULTS.md').write_text(text)
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        for arm,name in NAMES.items():ax.plot(range(101,106),[scores[k,arm].mean() for k in range(101,106)],marker='o',label=name)
        ax.set(xlabel='Training continuation seed',ylabel='Full-horizon EIG (nats)',title='Exploratory V4 replication');ax.legend();fig.tight_layout();fig.savefig(R/'five_seed_EIG.png',dpi=200);plt.close(fig)


def controller():
    global STOP
    lock=(O/'.controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True))
    signal.signal(signal.SIGINT,lambda *args:globals().__setitem__('STOP',True))
    assert read(O/'SEED_SMOKE.json')['status']=='PASS';verify_frozen()
    receipt=O/'CONTROLLER_LAUNCH.json'
    if receipt.exists():
        with (O/'CONTROLLER_RESUMES.jsonl').open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),time=time.time()))+'\n')
    else:put(receipt,dict(pid=os.getpid(),started=time.time(),queue=cfg()['queue'],study_type=STUDY))
    done=[];failed=[]
    for i,job in enumerate(cfg()['queue']):
        if STOP:break
        destination=O/job['output'];result=destination/'RESULTS.json'
        if result.exists() and read(result).get('status')=='COMPLETE' and (job['task']!='train' or (destination/'REPLICATION_VALIDATION.json').exists()):done.append(dict(**job,status='REUSED_COMPLETE'));continue
        if destination.exists():failed.append(dict(**job,error='Existing partial artifacts preserved;no silent retry'));continue
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            if STOP:break
            put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB',next=job,completed=done,failed=failed));time.sleep(15)
        if STOP:break
        args=['--task',job['task'],'--seed',str(job['seed']),'--arm',job['arm']]
        seconds=14400 if job['task']=='train' else 1800
        log=O/(job['id']+'.log')
        if log.exists():failed.append(dict(**job,error='Previous attempt log preserved;no silent retry'));continue
        with log.open('x') as f:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(seconds-30),sys.executable,'-u',str(SELF),*args],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),worker_pid=child.pid,job=job,started=time.time()))
            put(O/'STATUS.json',dict(status='RUNNING',current=job,pending=cfg()['queue'][i+1:],completed=done,failed=failed))
            while child.poll() is None:
                if STOP:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(1)
            code=child.wait()
        if code:failed.append(dict(**job,returncode=code))
        else:done.append(dict(**job,status='COMPLETE'))
        report()
    report();put(O/'STATUS.json',dict(status='TERMINATED_BY_USER' if STOP else 'COMPLETE' if not failed else 'FINISHED_WITH_FAILURES',completed=done,failed=failed,study_type=STUDY))


CURRENT_SEED=None
def main():
    global CURRENT_SEED
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['prepare','smoke','controller','train','eval101','secondary','report'],required=True)
    parser.add_argument('--seed',type=int,default=101);parser.add_argument('--arm',default='learned',choices=['learned','uniform','previous_learned','previous_uniform'])
    a=parser.parse_args();CURRENT_SEED=a.seed
    if a.task=='prepare':prepare()
    elif a.task=='smoke':smoke()
    elif a.task=='controller':controller()
    elif a.task=='train':train(a.seed,a.arm)
    elif a.task=='eval101':eval101(a.arm)
    elif a.task=='secondary':secondary(a.arm)
    elif a.task=='report':report()


if __name__=='__main__':main()
