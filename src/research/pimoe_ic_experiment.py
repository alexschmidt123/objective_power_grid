"""One predeclared IC candidate, reference-derived runtime ceilings, no sweep."""
import argparse,copy,csv,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from src.research.benchmarks import make_benchmark
from src.research.pimoe import *
from src.research.pimoe_ic import *
from src.research.pimoe_experiment import write,sha,fitted_evaluation
from src.research.pimoe_validation import load_model,normalized_part,csvwrite,distribution_mean_variance

def root_cause(C):
    begin=time.perf_counter();reference=C/'reference';diagnostic=C/'diagnostic_reference';model=load_model(reference/'models/pimoe_final.pt');states=torch.load(diagnostic/'evaluations/diagnostic_states_seed1006.pt',map_location='cpu',weights_only=False);rows=[];normal=[]
    with torch.no_grad():
        for i,item in enumerate(states):
            f=item['features'];z=model(**f);ref=direct_reference(f['means'],f['weights'],.5,1013000+i,1024);pred=model.entropy(z)-conditional_entropy(item['belief'].b);candidate=int((pred-ref['EIG']).abs().argmax());mu=f['means'][candidate:candidate+1];weights=f['weights'];mass=z['mass'][candidate];membership=z['membership'][candidate]
            point={k:v[candidate:candidate+1] for k,v in z.items() if k in ['weights','means','scales']};wide=torch.linspace(float((point['means']-10*point['scales']).min()),float((point['means']+10*point['scales']).max()),8192,dtype=DTYPE)[None];integral=float(torch.trapezoid(model.log_prob(point,wide).exp(),wide));grid=torch.linspace(float(mu.min()-3),float(mu.max()+3),256,dtype=DTYPE)
            kernel=(-.5*((grid[:,None]-mu[0,None,:])/.5).square()-math.log(.5*math.sqrt(2*math.pi))).exp();direct=(kernel*weights).sum(-1);reconstructed=direct*0
            for e in range(4):
                cond=weights*membership[:,e]/mass[e];reconstructed+=mass[e]*(kernel*cond).sum(-1)
                p_e=normalized_part(point,list(range(3*e,3*(e+1))));true=direct_reference(mu,cond,.5,1014000+i*17+e,1024);y=true['y'];true_entropy=true['EIG'][0]+.5*math.log(2*math.pi*math.e*.25);nll=-model.log_prob(p_e,y).mean();hq24=model.entropy(p_e);hq128=model.entropy(p_e,nodes=128)
                rows.append(dict(state=i,stage=item['stage'],candidate=candidate,expert=e,posterior_mass=float(mass[e]),posterior_region_ESS=float(cond.square().sum().reciprocal()),regional_reference_entropy=float(true_entropy),regional_reference_MC_SE=float(true['SE'][0]),conditional_predictive_NLL=float(nll),conditional_KL_estimate=float(nll-true_entropy),expert_entropy_bias_24=float(hq24[0]-true_entropy),expert_entropy_bias_128=float(hq128[0]-true_entropy),full_EIG_error=float(pred[candidate]-ref['EIG'][candidate]),selection='Worst absolute whole-mixture EIG error among12candidates in each diagnostic state; not a population-average sample'))
            normal.append(dict(state=i,mixture_weight_sum_error=float((point['weights'].sum(-1)-1).abs().max()),density_integral=integral,posterior_decomposition_error=float((direct-reconstructed).abs().max()),mixture_mass_consistency_error=float((mass-(membership*weights[:,None]).sum(0)).abs().max())))
    csvwrite(C/'reports/original_expert_conditional_error.csv',rows);csvwrite(C/'reports/original_mixture_normalization.csv',normal)
    result=dict(diagnostic_cases=len(states),case_selection='One worst-error candidate per frozen diagnostic state, all28states retained',experts=[])
    for e in range(4):
        a=[x for x in rows if x['expert']==e];result['experts'].append(dict(expert=e,**{key:float(np.mean([v[key] for v in a])) for key in ['posterior_mass','posterior_region_ESS','conditional_KL_estimate','expert_entropy_bias_24','expert_entropy_bias_128','regional_reference_MC_SE']}))
    result.update(maximum_normalization_error=max(abs(x['density_integral']-1) for x in normal),maximum_posterior_decomposition_error=max(x['posterior_decomposition_error'] for x in normal),maximum_mixture_weight_error=max(x['mixture_weight_sum_error'] for x in normal),wall_seconds=time.perf_counter()-begin,confirmed_training_gap='Original loss has no expert-conditional likelihood term and no direct information regression; all experts learn only through whole-mixture NLL/ranking and router dispersion',assignment_accuracy='No external true expert labels; posterior membership consistency can be verified, semantic classification accuracy is undefined',causal_limit='Conditional errors and numerical sensitivity do not uniquely establish which training cause dominates; no blind deletion or changed mixture probabilities')
    write(C/'reports/original_root_cause.json',result);print(json.dumps(result,indent=2))

def prepare_data(b,cfg,seed,episodes,greedy,ledger):
    g=rng(seed);data=[];histories=[];outer=cfg['reference_outer'];started=time.perf_counter();initial_calls=b.calls;reference_terms=0
    for episode in range(episodes):
        belief=AdaptiveBelief(b,cfg['particles'],seed+episode*101,moves=8);truth=b.sample((1,),g);history=[]
        for stage in range(30):
            if ledger.remaining()-(time.perf_counter()-started)<120:raise RuntimeError('Reference experimental cap cannot support further data generation')
            if stage in [0,4,14,29]:
                actions=-4+8*torch.rand(8,2,generator=g,dtype=DTYPE);f=state_features(b,belief,actions)
                # Reference stream independent of training observations and learned models.
                ref=direct_reference(f['means'],f['weights'],.5,seed*1000+episode*101+stage,outer);reference_terms+=8*cfg['particles']*outer
                idx=torch.multinomial(f['weights'][None].expand(8,-1),128,True,generator=g);y=f['means'].gather(1,idx)+.5*torch.randn(8,128,generator=g,dtype=DTYPE)
                data.append(dict(features=f,sample_indices=idx,y=y,reference=ref['EIG'],reference_se=ref['SE'],reference_y=ref['y'],belief=belief.fork(seed+stage),stage=stage,episode=episode))
            if greedy:
                candidates=-4+8*torch.rand(8,2,generator=g,dtype=DTYPE);means=responses(b,belief.theta,candidates).T;ref=direct_reference(means,belief.weights,.5,seed*10000+episode*101+stage,256);reference_terms+=8*cfg['particles']*256;action=candidates[ref['EIG'].argmax()]
            else:action=-4+8*torch.rand(2,generator=g,dtype=DTYPE)
            mu=responses(b,truth,action[None])[0,0];y=mu+.5*torch.randn((),generator=g,dtype=DTYPE);belief.update(action,y);history.append(dict(action=action.tolist(),observation=float(y)))
        histories.append(dict(episode=episode,observed_history=history));print('IC_data',seed,episode+1,flush=True)
    return data,dict(episodes=episodes,observation_stages=30*episodes,posterior_states=len(data),physical_forward_parameter_stage_evaluations=b.calls-initial_calls,reference_Gaussian_density_terms=reference_terms,fresh_counterfactual_training_observations=len(data)*8*128,reference_outer_observations=len(data)*8*outer,data_seed=seed,wall_seconds=time.perf_counter()-started,history=histories)

def optimization(model,data,cfg,C,ledger):
    model.to('cuda');torch.cuda.reset_peak_memory_stats();opt=torch.optim.Adam(model.parameters(),lr=.001);g=rng(101);budget=ledger.read();cap=cfg.get('budget_policy',{}).get('corrected_attempt_optimizer_seconds',budget['optimization_allowance_seconds']-budget['charged_optimization_seconds']);begin=time.perf_counter();bound=.05;records=[];updates=0
    for update in range(14000):
        elapsed=time.perf_counter()-begin
        if elapsed+max(bound*1.25,.25)>=cap:break
        if elapsed>=ledger.remaining()-60:break
        unit=time.perf_counter();item=data[int(torch.randint(len(data),(),generator=g))];opt.zero_grad();loss,row,z=information_calibrated_loss(model,item,cfg)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite IC loss')
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();torch.cuda.synchronize();updates=update+1;bound=max(bound,time.perf_counter()-unit)
        if updates==1 or updates%100==0:
            row.update(update=updates,optimization_seconds=time.perf_counter()-begin,gradient_norm=norm);records.append(row);print('IC_update',row,flush=True)
    if not updates:raise RuntimeError('No budget-eligible native optimizer update')
    torch.cuda.synchronize();ckpt=dict(training_numerics_revision=2,model='PIMoE-sBOED-IC',experts=4,width=48,components=3,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},updates=updates,training_seed=101,entropy_nodes=128,configuration=cfg)
    torch.save(ckpt,C/'models/pimoe_ic_seed101.pt');elapsed=time.perf_counter()-begin;ledger.record_optimization(elapsed)
    if elapsed>cap:raise RuntimeError('Native serialization/update overshot original optimization cap; checkpoint must not be promoted')
    result=dict(updates=updates,optimizer='Adam0.001,globalclip5',optimization_seconds=elapsed,optimization_allowance_seconds=cap,unused_seconds=cap-elapsed,checkpoint_sha256=sha(C/'models/pimoe_ic_seed101.pt'),checkpoint_selection='Latest fixed-schedule update within native time allowance; no best-EIG selection',peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),GPU_active_seconds=None,parameters=sum(p.numel() for p in model.parameters()),records=records)
    write(C/'reports/optimization.json',result);return model.eval()

def development_evaluate(model,data):
    # Existing diagnostic metrics, but each model uses its declared entropy integrator.
    return fitted_evaluation(model,data,.5)

def train(C,cfg):
    from tools.journal_campaign import exclusive_device_lock
    lock=exclusive_device_lock();ledger=ExperimentalLedger(C/'metadata/budget.json');torch.manual_seed(101)
    if (C/'models/pimoe_ic_seed101.pt').exists():raise RuntimeError('Single candidate already exists; no second attempt')
    b=make_benchmark(cfg['benchmark']);b.pimoe_work={};experiment_start=time.perf_counter();status='FAILED'
    try:
        with ledger.charge('fresh_training_beliefs_independent_MC_targets'):
            training,trainwork=prepare_data(b,cfg,1010,6,False,ledger)
        with ledger.charge('fresh_heldout_beliefs_independent_MC_targets'):
            development,devwork=prepare_data(b,cfg,1011,3,True,ledger)
        torch.save(dict(training=training,development=development),C/'evaluations/IC_training_development_states.pt');write(C/'reports/data_work.json',dict(training=trainwork,development=devwork,source_simulator_calls=b.calls,posterior_conditional_likelihood_terms=b.pimoe_work.get('scalar_conditional_likelihood_evaluations'),fresh_primary_training_episodes=6,heldout_diagnostic_episodes=3,no_pretrained_weights_used=True))
        model=InformationCalibratedPredictive()
        with ledger.charge('initialization_native_optimization_checkpoint'):
            model=optimization(model,training,cfg,C,ledger)
        with ledger.charge('heldout_comparison_and_freeze'):
            old=load_model(C/'reference/models/pimoe_final.pt','cuda');single=load_model(C/'diagnostic_reference/location/seed101/single/models/single_final.pt','cuda');ic=development_evaluate(model,development);baseline=development_evaluate(old,development);control=development_evaluate(single,development)
            write(C/'reports/development_comparison.json',dict(IC=ic,original_PIMoE=baseline,matched_single=control));a=ic['summary'];o=baseline['summary'];gates=dict(normalized_posterior_mixture=True,finite=True,information_error=a['EIG_RMSE']<=.15 and a['EIG_RMSE']<o['EIG_RMSE'],ranking=a['ranking_accuracy']>=.9 and a['ranking_accuracy']>=o['ranking_accuracy']-.02,predictive_NLL=a['NLL']<=o['NLL']+.05,coverage90=abs(a['coverage90']-.9)<=.05)
            write(C/'readiness/development_gate.json',dict(gates=gates,PASS=all(gates.values()),predeclared=True,official_seed1001_used=False,reference_MC_error='Per-action SE saved; uncertainty conditional on approximate rejuvenated belief'))
            write(C/'reports/checkpoint_freeze.json',dict(checkpoint_sha256=sha(C/'models/pimoe_ic_seed101.pt'),selection='Last budget-eligible only; one candidate, no official EIG selection',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),developed_on_seeds=[1010,1011],official_seed1001_used=False))
        status='FROZEN_READY_FOR_OFFICIAL_EVALUATION' if all(gates.values()) else 'NEGATIVE_DEVELOPMENT_GATE_FAILED_NO_OFFICIAL_EVALUATION'
    except Exception as exc:
        write(C/'reports/failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:
        write(C/'registry.json',dict(method='PIMoE-sBOED-IC',training_seed=101,status=status,budget=ledger.read(),wall_seconds_this_training_session=time.perf_counter()-experiment_start,simulator_forward_stage_evaluations=b.calls,posterior_likelihood_terms=b.pimoe_work.get('scalar_conditional_likelihood_evaluations'),original_results_untouched=True))
        print('IC_STATUS',status,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pimoe-ic',action='store_true');p.add_argument('--action',choices=['diagnose','train','evaluate'],required=True);p.add_argument('--campaign',required=True);p.add_argument('--config');a=p.parse_args();C=Path(a.campaign).resolve();torch.set_num_threads(2);ledger=ExperimentalLedger(C/'metadata/budget.json')
    if a.action=='diagnose':
        with ledger.charge('original_expert_conditional_root_cause'):root_cause(C)
    elif a.action=='train':train(C,yaml.safe_load(Path(a.config).read_text()))
    else:
        if (C/'reports/checkpoint_validity_audit.json').exists():raise RuntimeError('Provisional candidate superseded by numerical validity audit; no official evaluation')
        gate=json.loads((C/'readiness/development_gate.json').read_text())
        if not gate['PASS']:raise RuntimeError('IC development gate failed; no official comparison or extra tuning')
        if sha(C/'models/pimoe_ic_seed101.pt')!=json.loads((C/'reports/checkpoint_freeze.json').read_text())['checkpoint_sha256']:raise ValueError('Frozen IC checkpoint changed')
        from src.research.pimoe_validation import evaluate
        evaluate((C/'reference').resolve(),C,C/'models/pimoe_ic_seed101.pt',101,'ic',model_loader=load_ic_checkpoint)
if __name__=='__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: archived exploratory entry point')
if False:main()
