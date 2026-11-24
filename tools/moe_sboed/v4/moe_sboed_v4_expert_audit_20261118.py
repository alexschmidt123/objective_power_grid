"""Frozen V4 expert competence audit and canonical Fixed/Random controls. No training."""
import argparse,copy,fcntl,hashlib,importlib.util,json,math,os,resource,subprocess,sys,time
from pathlib import Path
M=Path.home()/'Documents/objective_power_grid'
O=M/'experiments/location_finding/moe_sboed_v4_expert_audit_20261118'
R=M/'reports/moe_sboed_v4_expert_audit_20261118'
A=M/'experiments/_legacy_archive/temp_research_20261115'
C=A/'scratch_runs/journal_LF_T30_native_final_20261102T052000Z'
B=A/'scratch_runs/moe_hec_v2_icml_20261109'
SELF=M/'tools/moe_sboed_v4_expert_audit_20261118.py'
PY=Path.home()/'miniconda3/envs/dellma/bin/python'
def read(p):return json.loads(Path(p).read_text())
def put(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8388608),b''):h.update(x)
    return h.hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def summary(x):
    import numpy as np
    a=np.asarray(x,dtype=float);return dict(mean=float(a.mean()),min=float(a.min()),max=float(a.max()),SD=float(a.std(ddof=1)) if a.size>1 else 0.)
def prepare():
    R.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for seed in range(101,106):
        cp=C/f'runs/seed{seed}/fixed/models/fixed.pt';receipt=cp.with_name('fixed_training.json');d=read(receipt)
        assert d.get('actual_updates',d.get('selected_update'))==50000
        jobs.append(dict(method='Fixed',seed=seed,checkpoint=str(cp),checkpoint_sha256=sha(cp),training_receipt=str(receipt),
          training_receipt_sha256=sha(receipt),training_seconds=d.get('total_offline_seconds',d.get('training_seconds')),
          seed_role='training_seed',compare_saved_reference=False))
        jobs.append(dict(method='Random',seed=seed,seed_role='policy_randomness_replicate;not_training_seed',
          distribution='truncated_standard_normal_in_[-4,4]^2',training_seconds=0,training_updates=0))
    protocol=dict(jobs=jobs,training_updates_this_campaign=0,architecture_changes=False,expert_checkpoints_changed=False,
      reference_table='reports/moe_sboed_v4_allmethods_20261117/RESULTS.json',
      canonical_bank_sha256='52c0c614c4ee3b63d58ccda60714b38b00a2c0ca35564a428fe8cc2ce8d727c5',
      source_sha256=sha(SELF),diagnostic_bank=dict(seed=202611180713,cases=32,contrasts=16384,
        purpose='New development-only mechanism audit;no checkpoint selection or architecture change'),
      diagnostic_scale=.25,diagnostic_scale_scope='Prespecified quarter-scale single-feature intervention only;not a corrected official baseline')
    assert not (R/'PROTOCOL.json').exists();put(R/'PROTOCOL.json',protocol);print(json.dumps(protocol,indent=2))

def baseline(method,seed):
    # Exactly the native adapter/import path used for the existing external baselines.
    ext=module('fixed_random_native_adapter',M/'tools/moe_sboed_external_evaluation.py');ext.prepare_imports()
    import numpy as np,torch
    from protocol import load,LOCK
    from src.research.journal_checkpoints import load_checkpoint
    from src.research.benchmarks import make_benchmark
    from src.research.engine import rollout
    from scipy.stats import t
    job=next(j for j in read(R/'PROTOCOL.json')['jobs'] if j['method']==method and j['seed']==seed)
    out=O/method/f'seed{seed}';assert not out.exists();out.mkdir(parents=True)
    held=[];start=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_SELF)
    try:
        for path in [M/'.gpu0.lock',Path.home()/'.cache/objective_power_grid/device_0.lock',Path.home()/'Documents/objective_power_grid_temp/.gpu0.lock']:
            f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        bank,u,scorer=load();assert LOCK['bank_sha256']==read(R/'PROTOCOL.json')['canonical_bank_sha256']
        raw=read(B/'inputs/HISTORICAL_BASELINES.json')['physics'];b=make_benchmark(raw)
        assert b.sigma==.5 and b.design_bounds==[-4.,4.] and b.random_design_distribution=='truncated_standard_normal'
        policy=None
        if method=='Fixed':
            assert sha(job['checkpoint'])==job['checkpoint_sha256'] and sha(job['training_receipt'])==job['training_receipt_sha256']
            policy=load_checkpoint(job['checkpoint'],b).cuda().eval();policy.requires_grad_(False)
            assert policy.kind=='fixed' and policy.horizon==30
            # Static schedule must not depend on the input history.
            with torch.no_grad():
                x=torch.zeros(2,121,device='cuda');y=torch.randn_like(x)
                for stage in range(30):assert torch.equal(policy(x,stage),policy(y,stage))
        values=[];curves=[];rollout_seconds=scoring_seconds=0.;samples=[];torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),torch.device('cuda'):
            negatives=torch.tensor(bank['contrasts'],device='cuda',dtype=torch.float64)
            for first in range(0,2000,16):
                assert time.monotonic()-start<14100
                # Fixed ignores RNG. Random gets independent policy RNG; test truth/noise stay identical.
                local_seed=2026110813902+first*97451+(seed-101)*1000000007
                torch.cuda.synchronize();tick=time.monotonic()
                data=rollout(b,policy,30,min(16,2000-first),local_seed,0,
                  truth=torch.tensor(bank['truth'][first:first+16],device='cuda',dtype=torch.float64),
                  noise=torch.tensor(bank['noise'][first:first+16,:,None],device='cuda',dtype=torch.float64),
                  method='random' if method=='Random' else None)
                assert all(torch.isfinite(a).all() and a.abs().max()<=4.000001 for a in data['actions'])
                torch.cuda.synchronize();rollout_seconds+=time.monotonic()-tick;tick=time.monotonic()
                score,_=scorer(b,data,negatives,16384,'cuda');torch.cuda.synchronize();scoring_seconds+=time.monotonic()-tick
                values.extend(score[:,-1].numpy());curves.append(score.numpy())
                put(out/'STATUS.json',dict(status='EVALUATING',method=method,seed=seed,completed=first+len(score),total=2000))
                if first%256==0:samples.append(dict(seconds=time.monotonic()-start,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True).strip()))
        values=np.asarray(values);np.save(out/'scores.npy',values);np.save(out/'curves.npy',np.concatenate(curves));se=float(values.std(ddof=1)/math.sqrt(2000));w=float(t.ppf(.975,1999))*se
        if method=='Fixed':assert sha(job['checkpoint'])==job['checkpoint_sha256']
        endcpu=resource.getrusage(resource.RUSAGE_SELF)
        put(out/'RESULTS.json',dict(status='COMPLETE',method=method,seed=seed,seed_role=job['seed_role'],EIG=float(values.mean()),MCSE=se,CI95=[float(values.mean()-w),float(values.mean()+w)],
          protocol=LOCK,source=job,scores_sha256=sha(out/'scores.npy'),training_updates_this_campaign=0,
          evaluation_seconds=time.monotonic()-start,rollout_seconds=rollout_seconds,scorer_seconds=scoring_seconds,
          online_amortized_episode_seconds=rollout_seconds/2000,online_scope='CUDA-synchronized batched rollout including history and simulator;not policy-only single-case latency',
          cpu_user_seconds=endcpu.ru_utime-cpu.ru_utime,cpu_system_seconds=endcpu.ru_stime-cpu.ru_stime,
          gpu_samples=samples,torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),simulator_observations=60000))
        put(out/'STATUS.json',dict(status='COMPLETE',method=method,seed=seed));print(json.dumps(read(out/'RESULTS.json')),flush=True)
    finally:
        for f in held:f.close()

def diagnostic(seed):
    bm=module('frozen_v4_allmethods',M/'tools/moe_sboed_v4_allmethods_20261117.py')
    import numpy as np,torch
    from torch import nn
    p=bm.p;engine=bm.engine;p.setup();out=O/f'audit_seed{seed}';assert not out.exists();out.mkdir(parents=True)
    with p.lease(),engine.Resources(out),torch.no_grad():
        state=p.load_parent('cuda');actor=bm.actor_from_template(state['algorithm'].actor_net,seed,'learned','cuda');del state
        cp=bm.checkpoint(seed,'learned');before=sha(cp)
        stored=np.load(p.n.BASE/'evaluation/diagnostic_candidate_histories.npz')['X'][:8]
        x=torch.tensor(stored.copy(),device='cuda').reshape(-1,117)
        z,raw,gates=actor.packets(x);full,info=actor.decode(x,z,gates);fused=info['routed']
        rms=lambda v:v.square().mean(-1).sqrt()
        basehidden=actor.action_head[0](fused);baseout=actor.action_head(fused)
        rows=[];cos=[]
        for i in range(4):
            for j in range(i+1,4):cos.append(dict(pair=[i+1,j+1],cosine=summary(torch.nn.functional.cosine_similarity(z[:,i],z[:,j],dim=-1).cpu())))
        for k in range(4):
            single=bm.ExtractedExpert(actor,k);actual=single(x);onehot=torch.zeros_like(gates);onehot[:,k]=1
            expected,_=actor.decode(x,z,onehot);torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            own=z[:,k];hidden=actor.action_head[0](own);output=actor.action_head(own)
            scaled=.25*own;scaled_action=(4*actor.action_head(scaled).tanh()+info['correction']).clamp(-4,4)
            rows.append(dict(expert=k+1,onehot_decode_max_error=float((actual-expected).abs().max()),
              expert_feature_RMS=summary(rms(own).cpu()),fused_feature_RMS=summary(rms(fused).cpu()),
              expert_to_fused_RMS_ratio=summary((rms(own)/rms(fused).clamp_min(1e-12)).cpu()),
              expert_to_fused_cosine=summary(torch.nn.functional.cosine_similarity(own,fused,dim=-1).cpu()),
              head_first_preactivation_RMS=summary(rms(hidden).cpu()),full_head_first_preactivation_RMS=summary(rms(basehidden).cpu()),
              action_RMS=summary(rms(actual).cpu()),full_action_RMS=summary(rms(full).cpu()),
              action_distance_from_full=summary((actual-full).norm(dim=-1).cpu()),
              near_boundary_coordinate_fraction=float((actual.abs()>3.9).double().mean()),
              full_near_boundary_coordinate_fraction=float((full.abs()>3.9).double().mean()),
              tanh_saturation_fraction=float((output.abs()>3).double().mean()),
              full_tanh_saturation_fraction=float((baseout.abs()>3).double().mean()),
              quarter_scale_action_distance_from_full=summary((scaled_action-full).norm(dim=-1).cpu())))
        class DiagnosticPolicy(nn.Module):
            def __init__(self,k=None,scale=1.):super().__init__();self.k=k;self.scale=scale
            def forward(self,x):
                if self.k is None:return actor(x)
                x=p.V4Actor.forward.__globals__['observed_prefix'](x)
                c=actor.query(x);own=actor.experts[self.k](c)[0]*self.scale
                return (4*actor.action_head(own).tanh()+.2*actor.shared_small(x).tanh()).clamp(-4,4)
        spec=read(R/'PROTOCOL.json')['diagnostic_bank'];rng=np.random.default_rng(spec['seed'])
        truth=rng.standard_normal((spec['cases'],4));noise=rng.standard_normal((spec['cases'],30));contrasts=rng.standard_normal((spec['contrasts'],4))
        np.savez(out/'diagnostic_bank.npz',truth=truth,noise=noise,contrasts=contrasts)
        banksha=sha(out/'diagnostic_bank.npz');lib=p.n.load_library();scorer=p.load_module('expert_audit_frozen_scorer',p.n.BASE/'source/frozen_evaluator.py').contrast_scores
        neg=torch.tensor(contrasts,device='cuda');rollouts=[]
        for label,k,scale in [('full',None,1.)]+[(f'e{k+1}',k,1.) for k in range(4)]+[(f'e{k+1}_quarter_scale_diagnostic',k,.25) for k in range(4)]:
            policy=DiagnosticPolicy(k,scale);th=torch.tensor(truth,device='cuda');eps=torch.tensor(noise,device='cuda')
            ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,dtype=torch.float64,device='cuda')
            for stage in range(30):
                xx=lib.packed(ds,ys,stage);action=policy(xx);assert torch.isfinite(action).all() and action.abs().max()<=4
                ds[:,stage]=action;ys[:,stage]=p.signal(th.reshape(-1,2,2),action.double())+.5*eps[:,stage]
            bench=type('Benchmark',(),{'sigma':.5,'calls':0})();scores,_=scorer(bench,dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th),neg,16384,'cuda')
            values=scores[:,-1].numpy();np.save(out/(label+'_scores.npy'),values);np.save(out/(label+'_actions.npy'),ds.cpu().numpy())
            rollouts.append(dict(mode=label,**engine.summaries(values),near_boundary_fraction=float((ds.abs()>3.9).double().mean()),
              mean_consecutive_design_distance=float((ds[:,1:]-ds[:,:-1]).norm(dim=-1).mean()),
              action_coordinate_std=ds.flatten(0,1).std(0).cpu().tolist()))
        logs=bm.v.training_path(seed,'learned')/'TRAINING_METRICS.jsonl';gradients={}
        for line in logs.read_text().splitlines():
            row=json.loads(line)
            for key,value in row.get('gradient_group_norms',{}).items():
                if key.startswith('experts.'):gradients.setdefault(key,[]).append(value)
        assert sha(cp)==before
        put(out/'RESULTS.json',dict(status='COMPLETE',seed=seed,checkpoint=str(cp),checkpoint_sha256=before,
          observed_histories=8,stages=30,same_history_tests=rows,pairwise_expert_cosines=cos,
          development_only_rollouts=rollouts,development_bank=spec,development_bank_sha256=banksha,
          joint_training_gradient_norms={k:summary(v) for k,v in gradients.items()},training_updates_this_job=0,
          scaling_intervention_scope='Diagnostic only;onehot weight1 remains the original extracted policy. Quarter scaling is not a bug fix,trained method or new official baseline.'))
        print(json.dumps(dict(seed=seed,status='COMPLETE',rollouts=rollouts)),flush=True)

def report():
    import numpy as np
    from scipy.stats import t
    old=read(M/'reports/moe_sboed_v4_allmethods_20261117/RESULTS.json');records=copy.deepcopy(old['records']);audit=[]
    for seed in range(101,106):
        path=O/f'audit_seed{seed}/RESULTS.json'
        if path.exists():audit.append(read(path))
        for name in ['Fixed','Random']:
            path=O/name/f'seed{seed}/RESULTS.json'
            if not path.exists():continue
            d=read(path);assert d['protocol']==old['protocol'];records.append(dict(method=name,seed=seed,EIG=d['EIG'],MCSE=d['MCSE'],result=str(path),scores=str(path.parent/'scores.npy'),seed_role=d['seed_role'],source=d['source']))
    names=list(old['summaries'])+['Fixed','Random'];summaries={};lookup={(r['method'],r['seed']):r for r in records}
    for name in names:
        values=[r['EIG'] for r in records if r['method']==name];summaries[name]=dict(n=len(values),mean=float(np.mean(values)) if values else None,SD=float(np.std(values,ddof=1)) if len(values)>1 else None)
    comparisons=copy.deepcopy(old['paired_seed_comparisons'])
    for name in ['Fixed','Random']:
        if summaries[name]['n']!=5:continue
        delta=[];within=[]
        for seed in range(101,106):
            x=np.load(lookup['moe-sboed',seed]['scores'])-np.load(lookup[name,seed]['scores']);delta.append(float(x.mean()));se=float(x.std(ddof=1)/math.sqrt(2000));w=float(t.ppf(.975,1999))*se
            within.append(dict(seed=seed,mean=float(x.mean()),MCSE=se,CI95=[float(x.mean()-w),float(x.mean()+w)]))
        mu=float(np.mean(delta));se=float(np.std(delta,ddof=1)/math.sqrt(5));w=float(t.ppf(.975,4))*se
        comparisons[name]=dict(mean_delta=mu,seed_SE=se,CI95=[mu-w,mu+w],within_replicate=within,
          p_two_sided=float(2*t.sf(abs(mu/se),4)) if se else float(mu==0),
          interpretation='Paired training seeds' if name=='Fixed' else 'Mixed learned-training and Random-policy replicate variation;not a five-training-seed Random comparison')
    # Recompute multiplicity over the expanded eleven-comparison family,not the old nine.
    if len(comparisons)==11:
        largest=0.
        for j,name in enumerate(sorted(comparisons,key=lambda n:comparisons[n]['p_two_sided'])):
            largest=max(largest,min(1.,(11-j)*comparisons[name]['p_two_sided']));comparisons[name]['p_Holm']=largest
            if 'seed_SE' in comparisons[name]:
                c=comparisons[name];w=float(t.ppf(1-.05/22,4))*c['seed_SE'];c['simultaneous_Bonferroni_CI95']=[c['mean_delta']-w,c['mean_delta']+w]
    status='COMPLETE' if len(records)==60 and len(audit)==5 else 'PENDING'
    put(R/'RESULTS.json',dict(status=status,completed_rows=len(records),required_rows=60,completed_expert_audits=len(audit),protocol=old['protocol'],records=records,summaries=summaries,comparisons=comparisons,
      original_comparison_report=str(M/'reports/moe_sboed_v4_allmethods_20261117/RESULTS.json'),
      Random_seed_scope='No training.101-105 index independent policy RNG streams on the same truth/noise bank.',training_updates_this_campaign=0))
    put(R/'EXPERT_DIAGNOSTICS.json',audit)
    text='# V4 moe-sboed: expert audit and Fixed/Random extension\n\nStatus: '+status+f'; {len(records)}/60 rows, {len(audit)}/5 expert audits. No retraining or checkpoint changes.\n\n'
    text+='|Method|101|102|103|104|105|Mean ± replicate SD|\n|---|---:|---:|---:|---:|---:|---:|\n';csvrows=[]
    for name in names:
        cells=[f"{lookup[name,s]['EIG']:.5f}" if (name,s) in lookup else 'PENDING' for s in range(101,106)];d=summaries[name];mu=f"{d['mean']:.5f} ± {d['SD']:.5f}" if d['n']==5 else 'PENDING'
        text+='|'+name+'|'+'|'.join(cells)+'|'+mu+'|\n';csvrows.append([name,*cells,mu])
    text+='\nRandom has no training; its columns denote policy-RNG replicates,not independently trained checkpoints. Fixed reuses five completed native 50,000-update checkpoints. All rows use the original canonical bank and frozen scorer. Shared pretraining and unequal historical compute remain disclosed in the original report.\n'
    (R/'RESULTS.md').write_text(text)
    import csv
    with (R/'MAIN_RESULTS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Method',101,102,103,104,105,'Mean ± replicate SD']);w.writerows(csvrows)
    return status

def controller():
    handle=(O/'.controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[dict(task='audit',seed=s) for s in range(101,106)]+[dict(task='baseline',method=m,seed=s) for m in ['Fixed','Random'] for s in range(101,106)]
    done=[]
    for job in jobs:
        target=O/(f"audit_seed{job['seed']}" if job['task']=='audit' else f"{job['method']}/seed{job['seed']}")
        if (target/'RESULTS.json').exists() and read(target/'RESULTS.json')['status']=='COMPLETE':done.append(job);continue
        assert not target.exists(),'Partial output preserved;manual review required'
        put(O/'STATUS.json',dict(status='RUNNING',current=job,completed=done,pending=jobs[len(done)+1:],training=False))
        command=[str(PY),'-B','-u',str(SELF),'--task',job['task'],'--seed',str(job['seed'])]
        if 'method' in job:command+=['--method',job['method']]
        with (O/(job['task']+'_'+job.get('method','')+f"_seed{job['seed']}.log")).open('x') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=M);put(O/'ACTIVE_PROCESS.json',dict(pid=child.pid,controller_pid=os.getpid(),job=job));code=child.wait()
        if code:put(O/'STATUS.json',dict(status='FAILED_PRESERVED',current=job,returncode=code,completed=done));report();return
        done.append(job);report()
    assert report()=='COMPLETE';put(O/'STATUS.json',dict(status='COMPLETE',completed=done,training=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['prepare','audit','baseline','controller','report'],required=True);p.add_argument('--seed',type=int);p.add_argument('--method',choices=['Fixed','Random']);a=p.parse_args()
    if a.task=='prepare':prepare()
    elif a.task=='audit':diagnostic(a.seed)
    elif a.task=='baseline':baseline(a.method,a.seed)
    elif a.task=='controller':controller()
    else:report()
if __name__=='__main__':main()
