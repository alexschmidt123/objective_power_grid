"""One seed, one four-hour candidate; frozen deployment and independent scoring."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import numpy as np
import torch
import yaml
from src.research.ir_moe import IRDeepSeekPolicy, integrated_spce, reward_to_go, load_ir
from src.research.deepseek_sboed import load_deepseek
from src.research.deepseek_sboed_v2 import load_v2
from src.research.deepseek_sboed_v3 import load_v3
from src.research.deepseek_sboed_experiment import write, sha, norm, save, diagnostics, feature_diagnostics
from src.research.deepseek_sboed_v2_experiment import paired, evaluate
from src.research.benchmarks import make_benchmark
from src.research.compute_budget import ComputeClock
from src.research.engine import rollout
from src.research.journal_validation import CommonValidation


def gradient_statistics(model, route_loss):
    # Offline diagnostic gradient of score loss only; retain graph for joint loss.
    parameters=list(model.router.parameters())
    grad=torch.autograd.grad(route_loss,parameters,retain_graph=True,allow_unused=True)
    vector=torch.cat([g.detach().flatten() for g in grad if g is not None])
    return vector


def train_ir(C,cfg,updates=50000,budget=14400,validator=None):
    out=C/'models/full';out.mkdir(parents=True,exist_ok=False)
    clock=ComputeClock(out/'compute_clock.json',budget,hardware=torch.cuda.get_device_name(0))
    torch.manual_seed(101)
    model=IRDeepSeekPolicy(balance_factor=cfg['balance_factor']).cuda().train()
    b=make_benchmark(cfg['benchmark'])
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,cfg['lr_decay'])
    baseline=torch.zeros(30,dtype=torch.float64,device='cuda')
    completed=0;bound=0.;records=[];curve=[];route_mean=None;route_m2=None;route_n=0
    stop='reference_schedule_completion'
    try:
        for update in range(1,updates+1):
            if not clock.can_start(max(10.,bound*1.5)):
                stop='compute_budget_exhaustion';break
            start=clock.spent();opt.zero_grad(set_to_none=True)
            with torch.device('cuda'):
                result=integrated_spce(b,model,30,cfg['batch_size'],101+104729*update,cfg['contrasts'],cfg['contrast_chunk'])
                info=result['info'][:,-1].mean()
                route,advantages=model.routing_loss(result['prefix_information'],baseline)
                balance=model.balance_loss()
                loss=-info+route+balance
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                log=update==1 or update%cfg.get('log_every',50)==0 or update==updates
                if log:
                    vector=gradient_statistics(model,route)
                    route_n+=1
                    if route_mean is None:route_mean=vector.clone();route_m2=torch.zeros_like(vector)
                    else:
                        delta=vector-route_mean;route_mean+=delta/route_n;route_m2+=delta*(vector-route_mean)
                    route_stats=dict(score_only_router_gradient_norm=float(vector.norm()),score_only_gradient_temporal_variance_trace=float(route_m2.sum()/max(1,route_n-1)),variance_samples=route_n,variance_definition='Across logged minibatch gradient estimates; includes parameter drift, not conditional fixed-policy variance',advantage_mean_by_stage=advantages.detach().mean(0).cpu().tolist(),advantage_variance_by_stage=advantages.detach().var(0).cpu().tolist(),baseline_used=baseline.cpu().tolist())
                if log: route_stats.update(model.score_variance(advantages))
                loss.backward();model.mask_unused_gradients()
                gradient=norm(model.parameters())
                if not math.isfinite(gradient):raise FloatingPointError('Nonfinite joint gradient')
                if log:
                    row=diagnostics(model,result,gradient,loss,info,update,clock.spent())
                    row.update(route_loss=float(route.detach()),balance_loss=float(balance.detach()),route_statistics=route_stats,contribution=model.last_contribution,preclip_shared_gradient=norm(model.shared.parameters()),preclip_router_gradient=norm(model.router.parameters()),preclip_expert_gradient=[norm(e.parameters()) for e in model.routed])
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize()
                # Past-update EMA, independent of all current expert draws.
                baseline.mul_(.95).add_(reward_to_go(result['prefix_information']).mean(0),alpha=.05)
            completed=update
            if (update-1)%cfg['lr_schedule_every']==0:scheduler.step()
            B,L=cfg['batch_size'],cfg['contrasts']
            for name,n in {'fresh_complete_primary_trajectories':B,'true_parameter_stages':B*30,'contrast_parameter_stages':B*L*30,'scalar_conditional_likelihood_evaluations':B*(L+1)*30,'policy_optimizer_updates':1,'shared_FFN_contexts':B*30,'routed_FFN_contexts':B*30*3}.items():clock.count(name,n)
            for name,n in result['work_counts'].items():clock.count(name,n)
            if log:
                row.update(learning_rate=opt.param_groups[0]['lr'],charged_seconds=clock.spent())
                records.append(row)
                write(C/'costs/full_progress.json',dict(records=records,clock=clock.report(),counts=model.counts()))
                print('IR_UPDATE',json.dumps(row),flush=True)
            # Frequent last snapshots reduce unused budget; preserve historical snapshots.
            if update==1 or update%cfg['checkpoint_every']==0 or update==updates:
                path=out/f'update{update}.pt';save(model,path,update,cfg,opt,scheduler);clock.checkpoint(str(path))
            if validator is not None and update%cfg['development_every']==0:
                with clock.exclude('validation'):
                    values,meta=validator(model);np.save(C/'evaluations'/f'development_update{update}.npy',values)
                    curve.append(dict(update=update,EIG=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values))),metadata=meta))
                    write(C/'reports/development_curve.json',curve)
                model.train()
            bound=max(bound,clock.spent()-start)
        if not completed:raise RuntimeError('No optimizer update')
        path=out/f'update{completed}_final.pt';save(model,path,completed,cfg,opt,scheduler);clock.checkpoint(str(path))
        selected=clock.selected();state=torch.load(selected['identifier'],map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);save(model,out/'selected.pt',state['update'],cfg)
        write(C/'costs/full_final.json',dict(stop_reason=stop,executed_updates=completed,selected_update=state['update'],checkpoint_sha256=sha(out/'selected.pt'),clock=clock.stop(),counts=model.counts(),forward_parameter_stages=b.calls,score_coefficient=1,baseline='previous-minibatch stage EMA .95',optimizer='Adam',GPU_active_seconds=None,disclosure='exclusive-device wall seconds, not GPU kernel busy time'))
        return model.eval()
    except Exception:
        write(C/'reports/training_failure.json',dict(traceback=traceback.format_exc(),updates=completed,clock=clock.stop()))
        raise


def gpu_correctness(cfg):
    torch.manual_seed(101);p=IRDeepSeekPolicy().cuda().eval();q=copy.deepcopy(p)
    with torch.device('cuda'):
        a=rollout(make_benchmark(cfg['benchmark']),p,30,4,400004,257)
        b=integrated_spce(make_benchmark(cfg['benchmark']),q,30,4,400004,257,64)
        torch.testing.assert_close(a['info'],b['prefix_information'],atol=1e-7,rtol=1e-7)
        (-a['info'][:,-1].mean()).backward();(-b['info'][:,-1].mean()).backward()
        for x,y in zip(p.parameters(),q.parameters()):
            assert (x.grad is None)==(y.grad is None)
            if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
    return dict(PASS=True,horizon=30,batch=4,contrasts=257,all_prefix_values_and_terminal_pathwise_gradients=True)


def prepare_confirmation(C,cfg):
    seed=cfg['confirmation_seed']
    folder=C/'evaluations/confirmation_reserved';folder.mkdir(exist_ok=False)
    v=CommonValidation(cfg['benchmark'],30,seed,2000,1000000,device='cpu')
    np.savez(folder/'cases.npz',truth=v.truth.numpy(),noise=v.noise.numpy());np.save(folder/'contrasts.npy',v.bank.numpy())
    write(folder/'manifest.json',dict(seed=seed,file_SHA256={p.name:sha(p) for p in [folder/'cases.npz',folder/'contrasts.npy']},reserved_before_training=True))


def development(C,cfg,model):
    v=CommonValidation(cfg['benchmark'],30,1031,256,262144,batch=16,chunk=16384,device='cuda:0')
    rows=[];arrays={}
    policies=[('learned',model)]
    for mode in ['uniform','mask_shared','mask_routed']:
        q=copy.deepcopy(model);q.intervention=mode;policies.append((mode,q))
    for label,p in policies:
        reps=3 if label=='uniform' else 1;values=[]
        for rep in range(reps):
            p._intervention_generator=None;p.intervention_seed=410101+rep
            x,meta=v(p);values.append(x)
        x=np.mean(values,0);arrays[label]=x;np.save(C/'evaluations'/f'development_{label}.npy',x)
        rows.append(dict(label=label,EIG=float(x.mean()),MC_SE=float(x.std(ddof=1)/math.sqrt(len(x))),paired_vs_learned=paired(x,arrays['learned']),metadata=meta,reps=reps,uniform_definition='Uniform random ordered subset, equal gates, preserved learned-top3 routed mass' if label=='uniform' else None))
    refs=[('v1',Path(cfg['reference_campaign'])/'models/full/selected.pt',load_deepseek),('v2',Path(cfg['v2_campaign'])/'models/full/selected.pt',load_v2),('dense_control',Path(cfg['v2_campaign'])/'models/dense/selected.pt',load_v2),('shared_only_control',Path(cfg['v2_campaign'])/'models/shared_only/selected.pt',load_v2),('routed_only_control',Path(cfg['v2_campaign'])/'models/routed_only/selected.pt',load_v2),('v3',Path(cfg['v3_campaign'])/'models/full/selected.pt',load_v3)]
    for label,path,loader in refs:
        if not path.exists():rows.append(dict(label=label,status='UNAVAILABLE',reason='No completed checkpoint'));continue
        p=loader(path,'cuda');x,meta=v(p);np.save(C/'evaluations'/f'development_{label}.npy',x)
        rows.append(dict(label=label,EIG=float(x.mean()),MC_SE=float(x.std(ddof=1)/math.sqrt(len(x))),paired_vs_IR=paired(arrays['learned'],x),metadata=meta,checkpoint_sha256=sha(path),contrast_budget_caveat='v1/v2/controls trained at lower contrast count; four-hour controls reused, not retrained'))
        del p
    write(C/'reports/development_comparisons.json',rows)
    feature_diagnostics(model,v,C)
    return rows


def fixed_history_swaps(C,cfg,model):
    """Nested prior-importance conditional predictive one-step IG; no hidden truth.

    Inference diagnostic only, not invoked by deployed policy. Limited histories
    and four expert subsets; report ESS and refuse unreliable histories.
    """
    from src.research.engine import replay_likelihood
    rng=torch.Generator(device='cuda').manual_seed(410107)
    b=make_benchmark(cfg['benchmark']);rows=[]
    with torch.no_grad(),torch.device('cuda'):
        result=rollout(b,model,30,4,410106,0)
        for stage in [1,5,15,29]:
            for episode in range(4):
                actions=[a[episode:episode+1] for a in result['actions'][:stage]]
                observations=[y[episode:episode+1] for y in result['observations'][:stage]]
                prior=b.sample((1,65536),rng);lp,_,_=replay_likelihood(b,prior,actions,observations)
                w=lp.softmax(-1)[0];ess=float(1/w.square().sum())
                base=dict(stage=stage,episode=episode,posterior_ESS=ess,prior_particles=65536)
                if ess<32:
                    rows.append(dict(**base,status='UNRELIABLE_POSTERIOR_NOT_SCORED'));continue
                f=result['features'][stage][episode:episode+1];z=model.encode_context(f,stage);scores=model.router(z).softmax(-1)
                learned=scores.topk(3,-1).indices
                sets=[learned]+[torch.randperm(7,generator=rng,device='cuda')[:3][None] for _ in range(3)]
                # Independent fantasy parameters from posterior weights, same noise
                # for all actions; full weighted bank supplies predictive denominator.
                idx=torch.multinomial(w,128,True,generator=rng);theta=prior[0,idx]
                eps=torch.randn(128,1,generator=rng,dtype=torch.float64)
                candidates=[]
                for ids in sets:
                    u=z+model.shared(z)+model.routed_feature_sum(z,scores.gather(1,ids),ids)
                    action=b.feasible(model.emitter(u),actions,30)
                    means,_=b.forward(prior,None,action[:,None].expand(-1,65536,-1))
                    truth_mean=means[0,idx];y=truth_mean+b.sigma*eps
                    ll=b.log_likelihood(y[:,None],means.expand(128,-1,-1))
                    numerator=b.log_likelihood(y,truth_mean)
                    gains=numerator-torch.logsumexp(ll+w.log()[None],-1)
                    candidates.append(dict(ids=ids[0].cpu().tolist(),action=action[0].cpu().tolist(),conditional_one_step_IG=float(gains.mean()),MC_SE=float(gains.std()/math.sqrt(128)),values=gains.cpu().numpy()))
                original=candidates[0]['values']
                for c in candidates:c['paired_difference_vs_learned']=paired(c.pop('values'),original)
                rows.append(dict(**base,status='FINITE_PARTICLE_APPROXIMATION_NOT_EXACT_POSTERIOR',candidates=candidates))
    write(C/'reports/fixed_history_swaps.json',dict(rows=rows,diagnostic_only=True,not_training_reward=True,warning='One-step conditional value, not full remaining-horizon EIG; ESS alone does not certify posterior convergence; no hidden simulated truth enters posterior or fantasies'))


def latency(C,cfg,model):
    b=make_benchmark(cfg['benchmark'])
    with torch.no_grad(),torch.device('cuda'):
        r=rollout(b,model,30,1,410110,0)
        samples=[]
        for rep in range(12):
            for t,f in enumerate(r['features']):
                torch.cuda.synchronize();start=time.perf_counter();model(f,t);torch.cuda.synchronize()
                if rep>=2:samples.append(time.perf_counter()-start)
    write(C/'reports/online_latency.json',dict(batch=1,device='RTX4090',decisions=len(samples),mean_ms=1000*float(np.mean(samples)),median_ms=1000*float(np.median(samples)),p95_ms=1000*float(np.quantile(samples,.95)),scope='neural decision including history encoder; explicit synchronization; excludes physical experiment and external history assembly',online_parameter_updates=0,online_searches=0))


def write_results(C,cfg):
    result=json.loads((C/'reports/ir_seed1001_result.json').read_text())
    confirmation=json.loads((C/'reports/ir_seed'+str(cfg['confirmation_seed'])+'_result.json').read_text())
    costs=json.loads((C/'costs/full_final.json').read_text())
    lat=json.loads((C/'reports/online_latency.json').read_text())
    dev=json.loads((C/'reports/development_comparisons.json').read_text())
    uniform=next(x for x in dev if x['label']=='uniform')
    refs={'vsOED':12.48085,'PIMoE-IC':11.88809,'Original PIMoE':11.41594,'MoE-v2':11.04023,'DeepSeekMoE-v1':10.98907,'Corrected DAD':10.27611,'Random':8.26624}
    text='# IR-DeepSeekMoE-sBOED seed101 results\n\n'
    text+=f"Official consumed benchmark seed1001: {result['EIG']:.8f} +/- {result['MC_SE']:.8f} nats (Monte Carlo SE), 2000 trajectories, 1000000 contrasts, T30.\n\n"
    text+=f"Reserved confirmation seed{cfg['confirmation_seed']}: {confirmation['EIG']:.8f} +/- {confirmation['MC_SE']:.8f} nats.\n\n"
    text+='| Method | T30 EIG nats | IR minus reference (unpaired unless separate array report) |\n|---|---:|---:|\n'
    for name,value in refs.items():text+=f'| {name} (reused historical mean) | {value:.5f} | {result["EIG"]-value:.5f} |\n'
    text+=f"\nTraining: {costs['clock']['charged_seconds']:.3f} charged seconds, {costs['executed_updates']} updates, selected update {costs['selected_update']}. Actual counters: `{costs['clock']['counts']}`. Readiness, development, diagnostics, official evaluations and latency costs are separate.\n\n"
    text+=f"Online neural latency: mean {lat['mean_ms']:.4f} ms, median {lat['median_ms']:.4f} ms.\n\n"
    text+='One training seed; no training-seed uncertainty estimate. Latest compute-eligible checkpoint was frozen before either official evaluation; no tuning followed. vsOED target aspirational. Matched dense comparison uses existing lower-contrast control and cannot isolate architecture from contrast-resolution/training-throughput differences.\n'
    (C/'reports/IR_MOE_SEED101_RESULTS.md').write_text(text)
    (C/'reports/IR_MOE_ROUTING_SPECIALIZATION.md').write_text('# Routing specialization\n\nFrozen paired development comparisons:\n\n```json\n'+json.dumps(dev,indent=2)+'\n```\n\nUniform-minus-learned paired difference and uncertainty are reported above. Useful information specialization is established only with a positive learned advantage supported by uncertainty and reliable context-dependent swap advantages; utilization/feature differences alone are insufficient. Fixed-history diagnostic is finite-particle one-step IG, not remaining-horizon value. See fixed_history_swaps.json including rejected low-ESS histories.\n')
    write(C/'reports/comparison.json',dict(IR=result,confirmation=confirmation,historical_differences={k:result['EIG']-v for k,v in refs.items()},uniform_minus_learned=uniform['paired_vs_learned'],one_training_seed=True))
    for name in ['IR_MOE_SEED101_RESULTS.md','IR_MOE_ROUTING_SPECIALIZATION.md']:
        (Path('reports')/name).write_text((C/'reports'/name).read_text())


def acquire(C):
    # Existing lock handles registered queues. Retry only when an unregistered
    # legitimate process is present; never preempt it or initialize CUDA first.
    from tools.journal_campaign import exclusive_device_lock
    while True:
        try:return exclusive_device_lock(wait=True)
        except RuntimeError as exc:
            if 'Device already has compute processes' not in str(exc):raise
            write(C/'registry.json',dict(status='QUEUED',reason=str(exc)));time.sleep(30)


def run(C,cfg,smoke=False):
    status='QUEUED';write(C/'registry.json',dict(status=status,PID=os.getpid(),main_allowance_seconds=14400))
    begin=time.monotonic();lock=acquire(C);queued=time.monotonic()-begin
    torch.set_num_threads(2)
    write(C/'run_config.json',dict(cfg,hardware=dict(execution_site='labpc',cluster=None,computer_name=platform.node(),cpu_model=platform.processor() or None,gpu_model=torch.cuda.get_device_name(0))))
    try:
        if smoke:
            start=time.monotonic();write(C/'readiness/gpu_equivalence.json',gpu_correctness(cfg))
            model=train_ir(C,cfg,updates=2,budget=600)
            write(C/'readiness/gpu_smoke.json',dict(PASS=True,seconds=time.monotonic()-start,batch=cfg['batch_size'],contrasts=cfg['contrasts'],horizon=30,updates=2,peak_allocated_bytes=torch.cuda.max_memory_allocated(),weights_reused=False))
            status='SMOKE_COMPLETE';return
        if not (C/'readiness/smoke_result.json').exists():raise RuntimeError('Missing passed native GPU smoke')
        if not json.loads((C/'readiness/smoke_result.json').read_text())['PASS']:raise RuntimeError('Native smoke failed')
        prepare_confirmation(C,cfg)
        v=CommonValidation(cfg['benchmark'],30,1030,128,262144,batch=16,chunk=16384,device='cuda:0')
        write(C/'registry.json',dict(status='MAIN_TRAINING',PID=os.getpid(),queue_seconds=queued))
        model=train_ir(C,cfg,validator=v)
        path=C/'models/full/selected.pt'
        if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Invalid final weights')
        write(C/'reports/checkpoint_freeze.json',dict(checkpoint_sha256=sha(path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),selection='Latest compute-eligible, no performance selection',deployment='deterministic top3',official_gate='numerical validity only',confirmation_seed=cfg['confirmation_seed']))
        write(C/'registry.json',dict(status='DEVELOPMENT_DIAGNOSTICS',PID=os.getpid()))
        start=time.monotonic();development(C,cfg,model)
        try: fixed_history_swaps(C,cfg,model)
        except Exception: write(C/'reports/optional_swap_failure.json',dict(traceback=traceback.format_exc()))
        latency(C,cfg,model)
        write(C/'costs/post_training_diagnostics.json',dict(seconds=time.monotonic()-start))
        write(C/'registry.json',dict(status='OFFICIAL_EVALUATION',PID=os.getpid()))
        official=evaluate(C,cfg,model,1001,'ir')
        original=np.load(Path(cfg['reference_campaign'])/'evaluations/official_seed1001_scores.npy')
        write(C/'reports/paired_official_vs_v1.json',paired(official,original))
        confirm=evaluate(C,cfg,model,cfg['confirmation_seed'],'ir')
        old=load_deepseek(Path(cfg['reference_campaign'])/'models/full/selected.pt','cuda')
        old_confirm=evaluate(C,cfg,old,cfg['confirmation_seed'],'v1')
        write(C/'reports/paired_confirmation_vs_v1.json',paired(confirm,old_confirm))
        write_results(C,cfg)
        status='COMPLETE_ONE_TRAINING_SEED'
    except Exception:
        write(C/'reports/failure.json',dict(traceback=traceback.format_exc()));status='FAILED_PRESERVED';raise
    finally:
        write(C/'registry.json',dict(status=status,PID=os.getpid(),total_wall_seconds=time.monotonic()-begin,queue_seconds=queued))
        print('IR_STATUS',status,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--ir-moe',action='store_true');p.add_argument('--campaign',required=True);p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(Path(a.campaign).resolve(),yaml.safe_load(Path(a.config).read_text()),a.smoke)
if __name__=='__main__':main()
