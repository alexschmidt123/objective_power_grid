
"""LabPC-only RC lifecycle; fresh training and frozen canonical evaluation."""
import argparse,copy,hashlib,json,math,re,subprocess,time,traceback
from pathlib import Path
import numpy as np,torch,yaml
from src.research.deepseek_rc import RoutedCalibrationPolicy
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.deepseek_sboed_experiment import write,sha,norm,save,feature_diagnostics
from src.research.deepseek_rc_train import train_rc
from src.research.engine import rollout
from src.research.benchmarks import make_benchmark
from src.research.journal_validation import CommonValidation
from src.research.deepseek_sboed_v2_experiment import evaluate,paired

def calibration(C,cfg):
    torch.manual_seed(101);m=RoutedCalibrationPolicy().cuda().eval();b=make_benchmark(cfg['benchmark'])
    begin=time.perf_counter()
    with torch.no_grad(),torch.device('cuda'):
        r=rollout(b,m,30,32,cfg['calibration_seed'],0)
        sn=[];rn=[]
        for t,f in enumerate(r['features']):
            z=m.encode_context(f,t);w,i=m.router(z).softmax(-1).topk(3,-1)
            sn.append(m.shared(z).square().mean());rn.append(DeepSeekPolicy.routed_feature_sum(m,z,w,i).square().mean())
        S=float(torch.stack(sn).mean().sqrt());R=float(torch.stack(rn).mean().sqrt())
    initial=min(3.,max(1.,math.sqrt(S/max(R,1e-12))))
    write(C/'readiness/calibration.json',dict(initial_gain=initial,shared_RMS=S,routed_RMS=R,seconds=time.perf_counter()-begin,parameter_stages=b.calls,seed=cfg['calibration_seed'],weights_discarded=True,official_data_used=False))
    return initial

def inspect_features(C,cfg,m,label):
    begin=time.perf_counter();m.eval();b=make_benchmark(cfg['benchmark'])
    with torch.no_grad(),torch.device('cuda'):r=rollout(b,m,30,64,cfg['final_development_seed']+11,0)
    rows=[]
    with torch.no_grad():
        for t,f in enumerate(r['features']):
            z=m.encode_context(f,t);p=m.router(z).softmax(-1);w,i=p.topk(3,-1)
            shared=m.shared(z);rr=DeepSeekPolicy.routed_feature_sum(m,z,w,i);base=m.emitter(z+shared+m.gain*rr)
            swap=i.clone();swap[:,-1]=p.topk(4,-1).indices[:,-1];alt=DeepSeekPolicy.routed_feature_sum(m,z,w,swap)
            physical=lambda u:8*u.sigmoid()-4
            rows.append(dict(stage=t,gain=float(m.gain),shared_norm=float(shared.norm(dim=-1).mean()),routed_norm=float(rr.norm(dim=-1).mean()),calibrated_routed_norm=float((m.gain*rr).norm(dim=-1).mean()),swap_design_RMS=float((physical(base)-physical(m.emitter(z+shared+m.gain*alt))).square().mean().sqrt()),mask_routed_design_RMS=float((physical(base)-physical(m.emitter(z+shared))).square().mean().sqrt()),mask_shared_design_RMS=float((physical(base)-physical(m.emitter(z+m.gain*rr))).square().mean().sqrt()),utilization=torch.nn.functional.one_hot(i,7).sum(1).double().mean(0).cpu().tolist(),entropy=float(-(p*p.log()).sum(-1).mean()),cosine=torch.einsum('bed,bfd->bef',torch.nn.functional.normalize(torch.stack([e(z) for e in m.routed],1),dim=-1),torch.nn.functional.normalize(torch.stack([e(z) for e in m.routed],1),dim=-1)).mean(0).cpu().tolist()))
    write(C/'reports'/f'{label}_contribution.json',dict(stages=rows,seconds=time.perf_counter()-begin,parameter_stages=b.calls,not_information_specialization_evidence=True))
    return rows

def smoke(C,cfg):
    gain=calibration(C,cfg);torch.manual_seed(101);m=RoutedCalibrationPolicy(initial_gain=gain).cuda().train();b=make_benchmark(cfg['benchmark']);o=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']));records=[];begin=time.perf_counter()
    for k in range(2):
        o.zero_grad(set_to_none=True)
        with torch.device('cuda'):
            r=rollout(b,m,30,2000,7126200+k,2000);loss=-r['info'][:,-1].mean()+m.balance_loss();loss.backward()
        pre={g:norm(p) for g,p in m.groups().items()}
        assert all(math.isfinite(x) and x>0 for x in pre.values()),pre
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True));post={g:norm(p) for g,p in m.groups().items()};o.step();torch.cuda.synchronize()
        assert torch.isfinite(torch.stack(r['actions'])).all() and torch.stack(r['actions']).abs().max()<=4
        records.append(dict(loss=float(loss),preclip=pre,postclip=post,global_preclip=gn,gain=float(m.gain),expert_gradients=[norm(e.parameters()) for e in m.routed]))
    rows=inspect_features(C,cfg,m,'smoke')
    assert max(x['swap_design_RMS'] for x in rows)>1e-9
    write(C/'readiness/GPU_smoke.json',dict(PASS=True,updates=2,batch=2000,contrasts=2000,horizon=30,seconds=time.perf_counter()-begin,parameter_stages=b.calls,records=records,weights_discarded=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))

def reserve(C,cfg):
    seed=cfg['confirmation_seed'];hits=[]
    for p in (Path.home()/'Documents').glob('objective_power_grid*/experiments/**/*'):
        if not p.is_file() or p.suffix not in ('.json','.yaml','.yml') or p.stat().st_size>2000000 or C in p.parents:continue
        try:
            if re.search(r'(?<![0-9.])'+str(seed)+r'(?![0-9.])',p.read_text()):hits.append(str(p))
        except (UnicodeError,OSError):pass
    if hits:raise ValueError('Confirmation seed already present: '+str(hits))
    v=CommonValidation(cfg['benchmark'],30,seed,2000,1000000,device='cpu')
    d=C/'evaluations/confirmation_reserved';d.mkdir(exist_ok=False)
    np.savez(d/'cases.npz',truth=v.truth.numpy(),noise=v.noise.numpy());np.save(d/'contrasts.npy',v.bank.numpy())
    write(d/'manifest.json',dict(seed=seed,file_SHA256={p.name:sha(p) for p in d.iterdir() if p.is_file()},prior_metadata_hits=hits,reserved_before_training=True))

def final(C,cfg,m):
    ck=C/'models/full/selected.pt';assert all(torch.isfinite(x).all() for x in m.state_dict().values())
    write(C/'reports/checkpoint_freeze.json',dict(checkpoint_sha256=sha(ck),gain=float(m.gain.detach()),selection='Latest eligible checkpoint',official_results_seen=False))
    dev=CommonValidation(cfg['benchmark'],30,cfg['final_development_seed'],256,262144,device='cuda:0')
    scores={};results={}
    for mode in ['full','uniform_weights','uniform_subset','mask_routed']:
        arrays=[]
        for rep in range(3 if mode=='uniform_subset' else 1):
            m.intervention=mode;m.intervention_seed=7126300+rep;m._intervention_generator=None
            a,meta=dev(m);arrays.append(a)
        v=np.mean(arrays,0);scores[mode]=v;np.save(C/'evaluations'/f'development_{mode}.npy',v)
        results[mode]=dict(EIG=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5),paired_vs_full=paired(v,scores['full']),metadata=meta)
    m.intervention='full';m._intervention_generator=None
    ref=Path(cfg['reference_campaign']);v1=load_deepseek(ref/'models/full/selected.pt','cuda');v,meta=dev(v1);np.save(C/'evaluations/development_v1.npy',v);results['v1']=dict(EIG=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5),RC_minus_v1=paired(scores['full'],v),metadata=meta)
    dense_path=Path.home()/'Documents/objective_power_grid_deepseekmoe_v2_20261023/experiments/10232026/10232026_deepseekmoe_sboed_v2/models/dense/selected.pt'
    if dense_path.exists():
        from src.research.deepseek_sboed_v2 import load_v2
        dense=load_v2(dense_path,'cuda');v,meta=dev(dense);np.save(C/'evaluations/development_dense.npy',v)
        results['dense']=dict(EIG=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5),RC_minus_dense=paired(scores['full'],v),metadata=meta,checkpoint_sha256=sha(dense_path),counts=dense.counts(),limitation='Approximate parameter match;50k reference schedule completed in12678s under14400s allowance; sameL2000; throughput implementation differs')
    write(C/'reports/development_comparisons.json',results);inspect_features(C,cfg,m,'final')
    timings=[]
    with torch.no_grad(),torch.device('cuda'):
        f=torch.zeros(1,121,dtype=torch.float64)
        for k in range(320):
            torch.cuda.synchronize();a=time.perf_counter();m(f,15);torch.cuda.synchronize()
            if k>=20:timings.append((time.perf_counter()-a)*1000)
    write(C/'reports/latency.json',dict(mean_ms=float(np.mean(timings)),p95_ms=float(np.quantile(timings,.95)),synchronized=True,decisions=300))
    official=evaluate(C,cfg,m,1001,'RC');evaluate(C,cfg,m,cfg['confirmation_seed'],'RC')
    assert sha(Path(cfg['canonical_evaluation_directory'])/'location_cases.npz')==cfg['cases_sha256']
    historical=np.load(ref/'evaluations/official_seed1001_scores.npy');write(C/'reports/paired_v1.json',paired(official,historical))
    assert sha(ck)==json.loads((C/'reports/checkpoint_freeze.json').read_text())['checkpoint_sha256']
    from src.research.deepseek_rc_report import finish_report,diagnostics
    diagnostics(C,cfg,m)
    finish_report(C,cfg)

def main():
    p=argparse.ArgumentParser();p.add_argument('--deepseek-rc',action='store_true');p.add_argument('--campaign',required=True);p.add_argument('--config',required=True);p.add_argument('--phase',choices=['smoke','main'],required=True);a=p.parse_args();C=Path(a.campaign).resolve();cfg=yaml.safe_load(Path(a.config).read_text())
    from tools.journal_campaign import exclusive_device_lock
    lock=exclusive_device_lock(wait=True);torch.set_num_threads(2);started=time.perf_counter()
    if a.phase=='smoke':smoke(C,cfg);return
    assert json.loads((C/'readiness/GPU_smoke.json').read_text())['PASS']
    assert 'passed' in (C/'readiness/CPU_tests.txt').read_text()
    reserve(C,cfg)
    for name,key in [('location_cases.npz','cases_sha256'),('location_contrasts.npy','contrast_sha256')]:
        assert sha(Path(cfg['canonical_evaluation_directory'])/name)==cfg[key]
    manifest=json.loads((C/'metadata/source_freeze.json').read_text())
    for name,h in manifest.items():assert sha(Path(name))==h,name
    try:
        write(C/'registry.json',dict(status='TRAINING',pid=__import__('os').getpid()))
        gain=json.loads((C/'readiness/calibration.json').read_text())['initial_gain']
        dev=CommonValidation(cfg['benchmark'],30,cfg['development_seed'],128,262144,device='cuda:0')
        m,stop=train_rc(C,cfg,'full',50000,14400,dev,policy_factory=lambda variant:RoutedCalibrationPolicy(variant=variant,initial_gain=gain))
        for name,h in manifest.items():assert sha(Path(name))==h,name
        write(C/'registry.json',dict(status='FROZEN_EVALUATION',stop=stop))
        final(C,cfg,m)
        write(C/'registry.json',dict(status='COMPLETE_ONE_SEED',campaign_wall_seconds=time.perf_counter()-started))
    except Exception:
        write(C/'registry.json',dict(status='FAILED_PRESERVED',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
