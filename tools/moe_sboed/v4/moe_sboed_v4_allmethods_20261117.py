"""V4 all-method canonical-bank evaluation. No training or checkpoint selection."""
import argparse, copy, csv, fcntl, importlib.util, json, math, os
from pathlib import Path
import subprocess, sys, time, traceback
M=Path.home()/'Documents/objective_power_grid'
O=M/'experiments/location_finding/moe_sboed_v4_allmethods_20261117'
R=M/'reports/moe_sboed_v4_allmethods_20261117'
SELF=M/'tools/moe_sboed_v4_allmethods_20261117.py'
spec=importlib.util.spec_from_file_location('v4_replication_readonly',M/'tools/moe_sboed_v4_replication_20261117.py')
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
p=v.p; engine=p.engine
import numpy as np
import torch
from torch import nn
from scipy.stats import t as student
METHODS=['moe-sboed','moe-sboed-uniform',*[f'moe-sboed-e{k}' for k in range(1,5)],'DAD','vsOED','RL-sBOED','ScoreBED']
SEEDS=list(range(101,106))
EXTERNAL={'DAD':'dad','vsOED':'vsoed','RL-sBOED':'rl_sboed','ScoreBED':'scorebed'}
FROZEN={}

def read(path):return json.loads(Path(path).read_text())
def put(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(Path(path),value)
def lock_protocol():
    v.verify_frozen();cert=engine.verify_references()
    return dict(architecture='V4 promoted to moe-sboed by user;historical artifacts unchanged',
      evaluation=engine.LOCK,training_updates=0,seeds=SEEDS,methods=METHODS,
      checkpoint_rule='Existing terminal V4 checkpoints only;no selection',
      expert_interface='One specialist plus copied common query provider,trained action head and weak correction;no router or other experts',
      expert_scope='Extracted inherited specialist policies;shared head was trained on mixtures,not independently optimized single policies',
      existing_replication='Exploratory continuation-seed study;new-bank results preserved separately',
      source_sha256={str(f.relative_to(M)):p.sha(f) for f in [SELF,M/'src/research/moe_sboed_v4_20261116/policy.py',M/'src/research/moe_sboed_v4_20261116/specialist_queries.py']},
      prior_historical_reproduction=cert['historical_reproduction'])

class ExtractedExpert(nn.Module):
    """Executable single specialist;not a masked execution of the ensemble."""
    def __init__(self,actor,k):
        super().__init__();self.index=k
        self.query=copy.deepcopy(actor.query)
        self.expert=copy.deepcopy(actor.experts[k])
        self.action_head=copy.deepcopy(actor.action_head)
        self.shared_small=copy.deepcopy(actor.shared_small)
        self.requires_grad_(False);self.eval()
    def forward(self,x):
        shape=x.shape[:-1];x=p.V4Actor.forward.__globals__['observed_prefix'](x).reshape(-1,117)
        z,_=self.expert(self.query(x))
        action=(4*self.action_head(z).tanh()+.2*self.shared_small(x).tanh()).clamp(-4,4)
        return action.reshape(*shape,2)

def checkpoint(seed,arm):return v.training_path(seed,arm)/'joint_actor.pt'
def actor_from_template(template,seed,arm,device):
    actor=v.new_actor(template,arm,seed).to(device)
    cp=checkpoint(seed,arm);actor.load_state_dict(torch.load(cp,map_location=device,weights_only=True)['actor_state'])
    actor.configure('frozen');actor.eval();return actor

@torch.no_grad()
def audit(actor,device):
    data=np.load(p.n.BASE/'evaluation/diagnostic_candidate_histories.npz')
    x=torch.tensor(data['X'][:2,[0,14,29]].copy(),device=device).reshape(-1,117)
    rows=[]
    for k in range(4):
        single=ExtractedExpert(actor,k)
        assert sum(isinstance(m,p.V4Actor.__init__.__globals__['V4Expert']) for m in single.modules())==1
        assert not hasattr(single,'router') and not hasattr(single,'experts')
        calls=[];hook=single.expert.register_forward_hook(lambda *args:calls.append(1))
        actual=single(x);hook.remove();assert len(calls)==1
        c=actor.query(x);z,_=actor.experts[k](c)
        expected=(4*actor.action_head(z).tanh()+.2*actor.shared_small(x).tanh()).clamp(-4,4)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        # The complete-model router and other experts must be irrelevant.
        handles=[]
        def forbidden(*unused):raise AssertionError('Forbidden expert/router execution')
        for module in [actor.router,*[e for j,e in enumerate(actor.experts) if j!=k]]:
            handles.append(module.register_forward_pre_hook(forbidden))
        try:torch.testing.assert_close(single(x),actual,atol=0,rtol=0)
        finally:
            for h in handles:h.remove()
        assert torch.isfinite(actual).all() and actual.abs().max()<=4
        rows.append(dict(method=METHODS[k+2],one_expert_only=True,router_absent=True,other_expert_calls=0,
          exact_single_path_error=float((actual-expected).abs().max()),finite_bounded_actions=True,
          diagnostic_stages=[0,14,29],parameters=sum(q.numel() for q in single.parameters())))
    # Check the uniform bypass on the same observed prefixes.
    uniform=copy.deepcopy(actor);uniform.arm='uniform';uniform.configure('frozen')
    handle=uniform.router.register_forward_pre_hook(lambda *unused:(_ for _ in ()).throw(AssertionError('uniform router executed')))
    try:
        _,_,g=uniform.packets(x);assert torch.equal(g,torch.full_like(g,.25))
    finally:handle.remove()
    return dict(status='PASS',paths=rows,uniform_router_bypassed=True,training_updates=0,
      limitation='Structural execution validity does not prove high EIG or independently trained expert competence.')

def smoke():
    protocol=lock_protocol();O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/'PROTOCOL.json').exists():assert read(R/'PROTOCOL.json')['evaluation']==protocol['evaluation']
    put(R/'PROTOCOL.json',protocol)
    state=p.load_parent('cpu');actor=actor_from_template(state['algorithm'].actor_net,101,'learned','cpu')
    result=audit(actor,'cpu');put(R/'STANDALONE_AUDIT.json',result);print(json.dumps(result))

def external_record(name,seed):
    folder=M/f'experiments/location_finding/moe_sboed_location_benchmark_20261115/external_evaluations/{EXTERNAL[name]}/seed{seed}'
    d=read(folder/'RESULTS.json');assert d['status']=='COMPLETE' and d['protocol']==engine.LOCK
    cp=d['checkpoint'];assert p.sha(Path(cp['checkpoint']))==cp['checkpoint_sha256']
    values=np.load(folder/'scores.npy');assert values.shape==(2000,) and np.isfinite(values).all()
    assert abs(float(values.mean())-d['mean'])<1e-12
    return dict(method=name,seed=seed,result=str(folder/'RESULTS.json'),scores=str(folder/'scores.npy'),
      scores_sha256=p.sha(folder/'scores.npy'),checkpoint=cp['checkpoint'],checkpoint_sha256=cp['checkpoint_sha256'],
      EIG=float(values.mean()),MCSE=float(values.std(ddof=1)/math.sqrt(2000)),training=cp,
      evaluation_seconds=d['evaluation_seconds'],online_amortized_seconds_per_episode=d['online_amortized_seconds_per_episode'],
      reused_completed_evaluation=True)

@torch.no_grad()
def worker(seed):
    lock_protocol();p.setup();root=O/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    with p.lease(),engine.Resources(root):
        deadline=time.monotonic()+14350
        state=p.load_parent('cuda');template=state['algorithm'].actor_net
        actor=actor_from_template(template,seed,'learned','cuda')
        uniform=actor_from_template(template,seed,'uniform','cuda');del state,template
        put(root/'STANDALONE_AUDIT.json',audit(actor,'cuda'))
        for name in METHODS[:6]:
            out=root/name
            if (out/'RESULTS.json').exists():
                d=read(out/'RESULTS.json');assert d['status']=='COMPLETE' and d['protocol']==engine.LOCK;continue
            assert not out.exists(),'Partial output preserved;manual review required: '+str(out)
            out.mkdir();start=time.monotonic()
            model=actor if name=='moe-sboed' else uniform if name=='moe-sboed-uniform' else ExtractedExpert(actor,int(name[-1])-1)
            arm='uniform' if name=='moe-sboed-uniform' else 'learned';cp=checkpoint(seed,arm);before=p.sha(cp)
            torch.cuda.reset_peak_memory_stats()
            values,result=engine.evaluate(model,out,name,deadline,collect=False)
            assert p.sha(cp)==before and result['protocol']==engine.LOCK
            if name.startswith('moe-sboed-e'):
                torch.save(dict(state_dict=model.state_dict(),expert=int(name[-1]),seed=seed,
                  source_checkpoint_sha256=before,architecture='V4 extracted single specialist with inherited head and common context'),out/'extracted_policy.pt')
            result.update(status='COMPLETE',method=name,seed=seed,architecture='V4',training_updates=0,
              checkpoint=str(cp),checkpoint_sha256=before,scores_sha256=p.sha(out/(name+'_scores.npy')),
              extracted=name.startswith('moe-sboed-e'),torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
            put(out/'RESULTS.json',result);put(out/'STATUS.json',dict(status='COMPLETE',method=name,seed=seed))
            print(json.dumps(dict(status='COMPLETE',method=name,seed=seed,EIG=result['mean'],seconds=time.monotonic()-start)),flush=True)
            report()
            if name.startswith('moe-sboed-e'):del model
        put(root/'STATUS.json',dict(status='COMPLETE',seed=seed,methods=METHODS[:6],training_updates=0))

def report():
    R.mkdir(parents=True,exist_ok=True);records=[];arrays={};inventory=[]
    for name in METHODS:
        for seed in SEEDS:
            if name in EXTERNAL:
                row=external_record(name,seed);values=np.load(row['scores'])
            else:
                folder=O/f'seed{seed}'/name;path=folder/'RESULTS.json'
                if not path.exists():continue
                d=read(path);assert d['status']=='COMPLETE' and d['protocol']==engine.LOCK
                scores=folder/(name+'_scores.npy');assert p.sha(scores)==d['scores_sha256']
                values=np.load(scores);assert values.shape==(2000,) and np.isfinite(values).all()
                row=dict(method=name,seed=seed,EIG=float(values.mean()),MCSE=float(values.std(ddof=1)/math.sqrt(2000)),
                  result=str(path),scores=str(scores),scores_sha256=d['scores_sha256'],checkpoint=d['checkpoint'],checkpoint_sha256=d['checkpoint_sha256'],
                  extracted=d['extracted'],evaluation_seconds=d['seconds'],online_amortized_seconds_per_episode=d['online_amortized_seconds_per_episode'],
                  torch_peak_allocated_bytes=d['torch_peak_allocated_bytes'],reused_completed_evaluation=False)
            records.append(row);arrays[name,seed]=values
    summaries={};comparisons={};within={}
    for name in METHODS:
        vals=[r['EIG'] for r in records if r['method']==name]
        summaries[name]=dict(n=len(vals),mean=float(np.mean(vals)) if vals else None,SD=float(np.std(vals,ddof=1)) if len(vals)>1 else None)
    for name in METHODS[1:]:
        pairs=[];cases=[]
        for seed in SEEDS:
            if ('moe-sboed',seed) not in arrays or (name,seed) not in arrays:continue
            delta=arrays['moe-sboed',seed]-arrays[name,seed];pairs.append(float(delta.mean()))
            cases.append(dict(seed=seed,**engine.summaries(delta)))
        if len(pairs)==5:
            mu=float(np.mean(pairs));sd=float(np.std(pairs,ddof=1));se=sd/math.sqrt(5);w=float(student.ppf(.975,4))*se
            simultaneous=float(student.ppf(1-.05/18,4))*se
            comparisons[name]=dict(mean_delta=mu,seed_SD=sd,seed_SE=se,CI95=[mu-w,mu+w],
              simultaneous_Bonferroni_CI95=[mu-simultaneous,mu+simultaneous],positive_seeds=sum(x>0 for x in pairs),
              p_two_sided=float(2*student.sf(abs(mu/se),4)) if se else float(mu==0),n=5)
        within[name]=cases
    if len(comparisons)==9:
        largest=0.
        for j,name in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['p_two_sided'])):
            largest=max(largest,min(1.,(9-j)*comparisons[name]['p_two_sided']));comparisons[name]['p_Holm']=largest
    status='COMPLETE' if len(records)==50 else 'PENDING_EVALUATIONS'
    caveats=['V4 is the proposed moe-sboed;historical V2/V3 remain archived.',
      'All scores in this report use the exact original canonical bank;new-bank V4 replication remains separate.',
      'Shared pretrained initialization:continuation-seed variability,not independent end-to-end training.',
      'E1-E4 execute only one specialist plus inherited common query provider,action head and weak correction;not independently trained policies.',
      'The shared head was optimized for mixtures;single-expert EIG measures extracted-path competence,not best achievable standalone training.',
      'Canonical bank informed historical development;this expanded benchmark is exploratory,not untouched confirmation.',
      'External native objectives/schedules and inherited pretraining differ;not equal-compute superiority.',
      'Evaluation MCSE conditions on one common contrast bank;finite-contrast bias is not included.']
    put(R/'RESULTS.json',dict(status=status,completed_rows=len(records),required_rows=50,protocol=engine.LOCK,
      records=records,summaries=summaries,paired_seed_comparisons=comparisons,paired_case_comparisons=within,caveats=caveats))
    put(R/'CHECKPOINT_MANIFEST.json',records)
    text='# moe-sboed: V4 all-method canonical-bank benchmark\n\nStatus: '+status+f' ({len(records)}/50 rows). No new training.\n\n'
    text+='|Method|101|102|103|104|105|Mean ± seed SD|\n|---|---:|---:|---:|---:|---:|---:|\n'
    lookup={(r['method'],r['seed']):r for r in records}
    csvrows=[]
    for name in METHODS:
        cells=[f"{lookup[name,s]['EIG']:.5f}" if (name,s) in lookup else 'PENDING' for s in SEEDS]
        d=summaries[name];summary=f"{d['mean']:.5f} ± {d['SD']:.5f}" if d['n']==5 else 'PENDING'
        text+='|'+name+'|'+'|'.join(cells)+'|'+summary+'|\n';csvrows.append([name,*cells,summary])
    text+='\n|moe-sboed minus|Mean paired difference|95% seed-level CI|Holm p|\n|---|---:|---|---:|\n'
    for name,d in comparisons.items():text+=f"|{name}|{d['mean_delta']:+.5f}|[{d['CI95'][0]:+.5f}, {d['CI95'][1]:+.5f}]|{d.get('p_Holm',float('nan')):.6g}|\n"
    text+='\n'+'\n'.join('- '+c for c in caveats)+'\n'
    (R/'RESULTS.md').write_text(text)
    with (R/'MAIN_RESULTS.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Method',*SEEDS,'Mean ± SD']);writer.writerows(csvrows)
    if status=='COMPLETE':
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5));xs=np.arange(10)
        ax.bar(xs,[summaries[m]['mean'] for m in METHODS],yerr=[summaries[m]['SD'] for m in METHODS],capsize=3)
        ax.set_xticks(xs,METHODS,rotation=35,ha='right');ax.set_ylabel('Full-horizon EIG (nats); bars: training-seed SD')
        fig.tight_layout();fig.savefig(R/'five_seed_comparison.png',dpi=180);plt.close(fig)
    return status

def controller():
    O.mkdir(parents=True,exist_ok=True);handle=(O/'.controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_protocol();assert read(R/'STANDALONE_AUDIT.json')['status']=='PASS';done=[]
    for seed in SEEDS:
        folder=O/f'seed{seed}'
        if (folder/'STATUS.json').exists() and read(folder/'STATUS.json')['status']=='COMPLETE':done.append(seed);continue
        assert not folder.exists(),'Preserve partial run;manual review required'
        put(O/'STATUS.json',dict(status='RUNNING',seed=seed,completed_seeds=done,pending_seeds=[s for s in SEEDS if s>seed],training_updates=0))
        with (O/f'seed{seed}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(SELF),'--task','worker','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,cwd=M)
            put(O/'ACTIVE_PROCESS.json',dict(controller_pid=os.getpid(),pid=child.pid,seed=seed,task='evaluation_only'))
            code=child.wait()
        if code:
            put(O/'STATUS.json',dict(status='FAILED_PRESERVED',seed=seed,returncode=code,completed_seeds=done));report();return
        done.append(seed);report()
    assert report()=='COMPLETE'
    put(O/'STATUS.json',dict(status='COMPLETE',completed_seeds=done,training_updates=0,completed_method_seed_rows=50))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['smoke','worker','controller','report'],required=True);parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.task=='smoke':smoke()
    elif args.task=='worker':worker(args.seed)
    elif args.task=='controller':controller()
    else:report()
if __name__=='__main__':main()
