"""Single v3 lifecycle using the existing launcher, device lock, trainer and evaluator."""
import argparse, copy, json, math, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from src.research.deepseek_sboed_v3 import DeepSeekV3Policy, streamed_spce
from src.research.deepseek_sboed import load_deepseek
from src.research.deepseek_sboed_experiment import train, write, sha, feature_diagnostics
from src.research.deepseek_sboed_v2_experiment import evaluate, paired, interventions
from src.research.journal_validation import CommonValidation
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout

def gpu_equivalence(cfg):
    torch.manual_seed(101)
    p=DeepSeekV3Policy().cuda();q=copy.deepcopy(p)
    with torch.device('cuda'):
        a=rollout(make_benchmark(cfg['benchmark']),p,30,4,400004,257)
        b=streamed_spce(make_benchmark(cfg['benchmark']),q,30,4,400004,257,64)
        torch.testing.assert_close(a['info'][:,-1],b['info'][:,-1],atol=1e-7,rtol=1e-7)
        (-a['info'][:,-1].mean()).backward();(-b['info'][:,-1].mean()).backward()
        maximum=0.
        for x,y in zip(p.parameters(),q.parameters()):
            assert (x.grad is None)==(y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
                maximum=max(maximum,float((x.grad-y.grad).abs().max()))
    return dict(PASS=True,max_gradient_difference=maximum,batch=4,contrasts=257,horizon=30)

def contribution_diagnostics(C,cfg,model):
    benchmark=make_benchmark(cfg['benchmark'])
    with torch.no_grad(),torch.device('cuda'):
        result=rollout(benchmark,model,30,32,400010,0)
        rows=[]
        for stage in range(30):
            z=model.encode_context(result['features'][stage],stage)
            shared=model.shared(z);scores=model.router(z).softmax(-1)
            weights,ids=scores.topk(model.top_k,-1)
            routed=model.routed_feature_sum(z,weights,ids)
            outputs=torch.stack([e(z) for e in model.routed],1)
            normalized=torch.nn.functional.normalize(outputs,dim=-1)
            rows.append(dict(stage=stage,shared_mean_norm=float(shared.norm(dim=-1).mean()),routed_sum_mean_norm=float(routed.norm(dim=-1).mean()),context_mean_norm=float(z.norm(dim=-1).mean()),mean_softmax=scores.mean(0).cpu().tolist(),activation_frequency=torch.nn.functional.one_hot(ids,7).sum(1).double().mean(0).cpu().tolist(),pairwise_feature_cosine=torch.einsum('bed,bfd->bef',normalized,normalized).mean(0).cpu().tolist()))
    write(C/'reports/contribution_diagnostics.json',dict(seed=400010,episodes=32,stages=rows,warning='Frozen predictive features; no automatic evidence of useful causal specialization'))

def run(C,cfg,wait=False):
    from tools.journal_campaign import exclusive_device_lock
    started=time.monotonic();status='QUEUED_WAITING_FOR_EXISTING_DEVICE_LOCK'
    write(C/'registry.json',dict(status=status,compute_budget_seconds=14400,training_started=False))
    print(status,flush=True)
    try:
        # Existing controller mechanism; no GPU context before lock acquisition.
        lock=exclusive_device_lock(wait=wait)
        queue_seconds=time.monotonic()-started
        readiness_started=time.monotonic()
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        begin=time.monotonic();check=gpu_equivalence(cfg)
        check['wall_seconds']=time.monotonic()-begin
        write(C/'readiness/GPU_objective_equivalence.json',check)
        smoke=C/'readiness/native_smoke';smoke.mkdir(exist_ok=False)
        for n in ['models','reports','costs','evaluations']:(smoke/n).mkdir()
        smoke_cfg=dict(cfg,checkpoint_every=1)
        trial,stop=train(smoke,smoke_cfg,'full',2,300,None,policy_factory=DeepSeekV3Policy)
        smoke_cost=json.loads((smoke/'costs/full_final.json').read_text())
        if smoke_cost['executed_updates']!=2 or stop!='reference_schedule_completion':
            raise RuntimeError('Native production-size smoke did not complete two updates')
        del trial;torch.cuda.empty_cache()
        write(C/'costs/development_readiness.json',dict(total_readiness_wall_seconds=time.monotonic()-readiness_started,queue_wait_seconds=queue_seconds,GPU_equivalence=check,native_smoke_training_seconds=smoke_cost['clock']['charged_seconds'],all_discarded_training_preserved=True))
        write(C/'readiness/native_smoke_result.json',dict(PASS=True,batch=2000,contrasts=8192,updates=2,weights_discarded=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),costs=smoke_cost))
        v0=CommonValidation(cfg['benchmark'],30,1030,128,262144,batch=16,chunk=16384,device='cuda:0')
        v1=CommonValidation(cfg['benchmark'],30,1031,256,262144,batch=16,chunk=16384,device='cuda:0')
        write(C/'registry.json',dict(status='MAIN_TRAINING',training_started=True,queue_wait_seconds=queue_seconds,main_cap_seconds=14400,development_costs_separate=True))
        model,stop=train(C,cfg,'full',50000,14400,v0,policy_factory=DeepSeekV3Policy)
        if stop not in ['reference_schedule_completion','compute_budget_exhaustion']:
            raise RuntimeError('Checkpoint failed numerical/stability criterion: '+stop)
        if not all(torch.isfinite(x).all() for x in model.parameters()):
            raise FloatingPointError('Nonfinite final parameters')
        reference=Path(cfg['reference_campaign']);development=[]
        for seed,validator,oldpath in [(1030,v0,reference/'evaluations/final_development_full.npy'),(1031,v1,Path(cfg['v2_campaign'])/'diagnostics/full_0.npy')]:
            values,metadata=validator(model);old=np.load(oldpath)
            if not np.isfinite(values).all():raise FloatingPointError('Invalid development scores')
            np.save(C/'evaluations'/f'v3_development{seed}.npy',values)
            development.append(dict(seed=seed,v3_mean=float(values.mean()),v1_mean=float(old.mean()),paired=paired(values,old),metadata=metadata))
        write(C/'reports/development_comparison.json',development)
        path=C/'models/full/selected.pt'
        write(C/'reports/checkpoint_freeze.json',dict(checkpoint_sha256=sha(path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),selection='Latest compute-eligible checkpoint, no development/official best selection',numerically_valid=True,official_performance_gate=False,confirmation_seed=cfg['confirmation_seed']))
        feature_diagnostics(model,v1,C);contribution_diagnostics(C,cfg,model);interventions(C,cfg,model,'v3',v1)
        historical=evaluate(C,cfg,model,1001,'v3')
        write(C/'reports/historical_paired_v3_minus_v1.json',paired(historical,np.load(reference/'evaluations/official_seed1001_scores.npy')))
        # Both final evaluations are fixed in advance. No training follows them.
        original=load_deepseek(reference/'models/full/selected.pt','cuda')
        fresh_old=evaluate(C,cfg,original,cfg['confirmation_seed'],'v1')
        fresh=evaluate(C,cfg,model,cfg['confirmation_seed'],'v3')
        write(C/'reports/confirmation_paired_v3_minus_v1.json',paired(fresh,fresh_old))
        cost=json.loads((C/'costs/full_final.json').read_text())
        result=json.loads((C/'reports/v3_seed1001_result.json').read_text())
        difference=json.loads((C/'reports/historical_paired_v3_minus_v1.json').read_text())
        text='# V3 seed101 completed\n\nOfficial historical EIG: '+str(result['EIG'])+' +/- '+str(result['MC_SE'])+' nats (evaluation MC SE).\n\nPaired difference versus v1: '+str(difference)+'\n\nTraining cost and stop: '+str(cost['clock']['charged_seconds'])+' seconds; '+cost['stop_reason']+'; executed updates '+str(cost['executed_updates'])+'.\n\nReserved independent confirmation and raw per-trajectory results are saved separately. One training seed only; no retuning after these evaluations.\n'
        (C/'reports/V3_SEED101_RESULTS.md').write_text(text)
        write(C/'reports/comparison_vs_historical_means.json',dict(EIG=result['EIG'],MC_SE=result['MC_SE'],differences={name:result['EIG']-value for name,value in {'DeepSeekMoE_v1':10.98906589858419,'DAD':10.276112861655836,'MoE_v2':11.04023,'vsOED':12.48085}.items()},paired_vs_v1=difference,other_differences_unpaired=True))
        status='COMPLETE_ONE_VALID_TRAINING_SEED'
    except Exception as exc:
        status='FAILED_PRESERVED'
        write(C/'reports/failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        write(C/'registry.json',dict(status=status,total_wall_seconds=time.monotonic()-started,training_cost_file=str(C/'costs/full_final.json'),historical_results_untouched=True))
        print('V3_STATUS',status,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--deepseek-sboed-v3',action='store_true');p.add_argument('--campaign',required=True);p.add_argument('--config',required=True);p.add_argument('--wait-for-device',action='store_true')
    a=p.parse_args();run(Path(a.campaign).resolve(),yaml.safe_load(Path(a.config).read_text()),a.wait_for_device)
if __name__=='__main__':main()
