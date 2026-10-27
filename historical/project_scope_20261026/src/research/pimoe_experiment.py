"""Bounded PIMoE readiness/training entry point, invoked only via run.sh --pimoe.
No separate scheduler. Failed scientific gates prevent expensive training/evaluation.
"""
import argparse,csv,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from src.research.benchmarks import make_benchmark
from src.research.pimoe import *

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dataset(b,cfg,seed,episodes,particles):
    g=rng(seed);data=[];histories=[]
    for episode in range(episodes):
        truth=b.sample((1,),g);belief=AdaptiveBelief(b,particles,seed+episode*101,moves=cfg['move_steps']);actions=[];ys=[]
        for stage in range(30):
            if time.monotonic()-cfg['development_start']>cfg['allowance_seconds']-180:raise RuntimeError('Whole-development allowance exhausted during data generation')
            if stage in [0,4,14,29]:
                lo,hi=b.design_bounds;a=lo+(hi-lo)*torch.rand(cfg['candidate_actions'],2,generator=g,dtype=DTYPE)
                features=state_features(b,belief,a)
                # Ranking references and predictive targets use independent latent/noise draws.
                ref_begin=time.monotonic();ref=direct_reference(features['means'],features['weights'],b.sigma,seed+episode*1009+stage*17,cfg['reference_outer'])
                b.pimoe_work['direct_reference_wall_seconds']+=time.monotonic()-ref_begin
                b.pimoe_work['scalar_conditional_likelihood_evaluations']+=cfg['candidate_actions']*particles*cfg['reference_outer']
                ids=torch.multinomial(belief.weights,cfg['predictive_samples'],True,generator=g)
                train_y=features['means'][:,ids]+b.sigma*torch.randn(cfg['candidate_actions'],cfg['predictive_samples'],generator=g,dtype=DTYPE)
                b.pimoe_work['counterfactual_observations']+=train_y.numel()+ref['y'].numel()
                data.append(dict(features=features,y=train_y,reference=ref['EIG'],reference_se=ref['SE'],reference_y=ref['y'],belief=belief.fork(seed+stage+episode),episode=episode,stage=stage))
            # Simulator-only exploratory histories, no trained action sub-policy.
            lo,hi=b.design_bounds;a=lo+(hi-lo)*torch.rand(1,2,generator=g,dtype=DTYPE)
            mu=responses(b,truth,a)[0,0];y=mu+b.sigma*torch.randn((),generator=g,dtype=DTYPE)
            belief.update(a[0],y);actions.append(a[0]);ys.append(y);b.pimoe_work['true_observation_stages']+=1
        b.pimoe_work['fresh_complete_primary_episodes']+=1
        histories.append(dict(actions=torch.stack(actions).tolist(),observations=torch.stack(ys).tolist(),posterior_diagnostics=belief.records))
        print('data_episode',seed,episode+1,flush=True)
    return data,histories

def fitted_evaluation(model,data,sigma):
    out=[];started=time.monotonic()
    with torch.no_grad():
        for item in data:
            z=model(**item['features']);pred=(model.entropy(z)-.5*math.log(2*math.pi*math.e*sigma**2)).cpu();reference=item['reference'];se=item['reference_se']
            y=item['y'].to(z['means']);lp=model.log_prob(z,y).cpu();pit=model.cdf(z,y).cpu().flatten().sort().values;uniform=(torch.arange(len(pit),dtype=DTYPE)+.5)/len(pit);ks=float((pit-uniform).abs().max());coverage=float(((pit>.05)&(pit<.95)).double().mean())
            diff=reference[:,None]-reference[None,:];valid=torch.triu(diff.abs()>2*(se[:,None].square()+se[None,:].square()).sqrt(),diagonal=1);correct=((pred[:,None]-pred[None,:]).sign()==diff.sign())
            pair_count=int(valid.sum());hit=int(correct[valid].sum());chosen=int(pred.argmax());regret=float(reference.max()-reference[chosen]);error=pred-reference
            # Direct non-neural posterior predictive log density on identical held-out y.
            f=item['features'];means=f['means'];weights=f['weights'];direct=-.5*((item['y'][:,:,None]-means[:,None,:])/sigma).square()-math.log(sigma*math.sqrt(2*math.pi));direct=torch.logsumexp(direct+weights.clamp_min(1e-300).log()[None,None],-1)
            out.append(dict(stage=item['stage'],NLL=float(-lp.mean()),reference_NLL=float(-direct.mean()),NLL_excess=float((direct-lp).mean()),EIG_RMSE=float(error.square().mean().sqrt()),EIG_bias=float(error.mean()),ranking_correct=hit,ranking_pairs=pair_count,ranking_accuracy=hit/pair_count if pair_count else None,KS=ks,coverage90=coverage,selected_action_reference_EIG=float(reference[chosen]),selected_action_regret=regret,expert_mass=z['mass'].mean(0).cpu().tolist()))
    pairs=sum(r['ranking_pairs'] for r in out);summary={k:float(np.mean([r[k] for r in out])) for k in ['NLL','reference_NLL','NLL_excess','EIG_RMSE','EIG_bias','KS','coverage90','selected_action_reference_EIG','selected_action_regret']};summary.update(ranking_pairs=pairs,ranking_accuracy=sum(r['ranking_correct'] for r in out)/pairs if pairs else None,wall_seconds=time.monotonic()-started)
    return dict(summary=summary,states=out)

def train_model(model,data,updates,cfg,campaign,start,recordname):
    model.to(cfg['device']);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);g=rng(cfg['training_seed']);records=[];began=time.monotonic();done=0
    for update in range(updates):
        if time.monotonic()-start>=cfg['allowance_seconds']-180:break
        item=data[int(torch.randint(len(data),(),generator=g))];optimizer.zero_grad();loss,metrics,_=predictive_loss(model,item['features'],item['y'],item['reference'],item['reference_se'])
        if not torch.isfinite(loss):raise FloatingPointError('Predictive loss not finite')
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));optimizer.step();done=update+1
        if done==1 or done%100==0:
            row=dict(update=done,loss=float(loss.detach()),gradient_norm=norm,charged_seconds=time.monotonic()-start,**metrics);records.append(row);print(recordname,row,flush=True)
    if torch.cuda.is_available():torch.cuda.synchronize()
    torch.save(dict(model='PIMoE-sBOED' if model.expert_count>1 else 'internal_single_density_control',state_dict={k:v.cpu() for k,v in model.state_dict().items()},experts=model.expert_count,width=model.experts[0].net[0].out_features,components=model.components,updates=done,charged_seconds=time.monotonic()-start),campaign/'models'/f'{recordname}.pt')
    write(campaign/'reports'/f'{recordname}_training.json',dict(updates=done,wall_seconds=time.monotonic()-began,parameters=sum(p.numel() for p in model.parameters()),records=records,objective='predictiveNLL+0.1 confidence-masked ranking+0.02 physical-response within-region variance'))
    return done

def posterior_validation(b,items,cfg):
    result=[]
    for item in items:
        if item['episode']!=0:continue
        old=item['belief'];ref=AdaptiveBelief(b,cfg['posterior_validation_particles'],5004+item['stage'],moves=cfg['move_steps']*2)
        for a,y in zip(old.actions,old.ys):ref.update(a,y)
        ref.refine(cfg['posterior_validation_particles'],steps=32)
        a=item['features']['actions'];f=state_features(b,ref,a);x=direct_reference(f['means'],f['weights'],b.sigma,6004+item['stage'],cfg['reference_outer']);y=item['reference'];se=(x['SE'].square()+item['reference_se'].square()).sqrt();delta=x['EIG']-y
        b.pimoe_work['scalar_conditional_likelihood_evaluations']+=a.shape[0]*len(ref.theta)*cfg['reference_outer']
        # 0.15 nat numerical agreement plus MC error; not proof of exact posterior.
        passes=bool((delta.abs()<=.15+2*se).all())
        result.append(dict(stage=item['stage'],particles=len(old.theta),independent_particles=len(ref.theta),maximum_abs_EIG_difference=float(delta.abs().max()),mean_abs_EIG_difference=float(delta.abs().mean()),maximum_combined_SE=float(se.max()),passes=passes,independent_rejuvenation=ref.records[-1]))
    return dict(passes=all(r['passes'] for r in result),states=result,limitation='Independent SMC convergence diagnostic, not exact-posterior certification; missing separated modes can remain')

def final_evaluate(model,b,cfg,campaign,start):
    # No training after this point. One common seed1001 bank, frozen checkpoint.
    canonical=Path(cfg['canonical_evaluation_directory']);cases=np.load(canonical/'location_cases.npz');bank=np.load(canonical/'location_contrasts.npy')
    if sha(canonical/'location_cases.npz')!=cfg['cases_sha256'] or sha(canonical/'location_contrasts.npy')!=cfg['contrast_sha256']:raise ValueError('Canonical bank identity mismatch')
    from src.research.engine import rollout
    from src.research.sir_evaluation import contrast_scores
    # The generic explicit-likelihood scorer handles location without ODE semantics.
    truth_key='truth' if 'truth' in cases else 'theta';noise_key='noise' if 'noise' in cases else 'measurement_noise_standard'
    truth=torch.tensor(cases[truth_key],dtype=DTYPE);noise=torch.tensor(cases[noise_key],dtype=DTYPE);bank=torch.tensor(bank,dtype=DTYPE);scores=[];latencies=[];begin=time.monotonic()
    for i in range(2000):
        planner=PIMoEPlanner(model,b,seed=9000000+i,particles=cfg['planner_particles'],two_step=True)
        actions=[];obs=[]
        for t in range(30):
            action=planner.select();mu=responses(b,truth[i:i+1],action[None])[0,0];y=mu+b.sigma*noise[i,t,0];actions.append(action[None]);obs.append(y.reshape(1,1));planner.observe(action,y)
        result=dict(actions=actions,observations=obs,truth=truth[i:i+1]);information,_=contrast_scores(b,result,bank,4096);scores.append(float(information[0,-1]));latencies+=planner.timings
        if (i+1)%10==0:print('final_evaluation',i+1,flush=True)
    np.save(campaign/'evaluations/seed1001_scores.npy',scores);write(campaign/'reports/final_evaluation.json',dict(EIG=float(np.mean(scores)),MC_SE=float(np.std(scores,ddof=1)/math.sqrt(len(scores))),systems=2000,contrasts=1000000,seed=1001,mean_online_decision_seconds=float(np.mean(latencies)),evaluation_seconds=time.monotonic()-begin))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pimoe',action='store_true');parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();C=Path(args.output).resolve();cfg=yaml.safe_load(Path(args.config).read_text());clock=json.loads((C/'reports/development_clock.json').read_text());start=clock['start_monotonic'];cfg['development_start']=start;torch.set_num_threads(2);torch.manual_seed(cfg['training_seed'])
    from tools.journal_campaign import exclusive_device_lock
    lock=exclusive_device_lock();cfg['device']='cuda' if torch.cuda.is_available() else 'cpu';b=make_benchmark(cfg['benchmark']);assert b.name=='location_finding' and b.design_bounds==[-4.,4.] and b.sigma==.5
    b.pimoe_work=dict(fresh_complete_primary_episodes=0,true_observation_stages=0,scalar_conditional_likelihood_evaluations=0,counterfactual_observations=0,direct_reference_wall_seconds=0.)
    write(C/'reports/run_metadata.json',dict(command=['bash','run.sh',*os.sys.argv[1:]],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',configuration_sha256=sha(args.config),benchmark_source_sha256=sha(Path(__file__).parent/'benchmarks.py'),benchmark=cfg['benchmark'],training_seed=101,development_seeds=[1004,1005],final_seed_used_for_tuning=False))
    status='FAILED';why=None
    try:
        train,hist=dataset(b,cfg,1004,cfg['training_episodes'],cfg['particles']);held,testhist=dataset(b,cfg,1005,cfg['validation_episodes'],cfg['particles']);write(C/'reports/exploratory_histories.json',dict(training=hist,heldout=testhist));torch.save(dict(training=train,heldout=held),C/'models/posterior_training_states.pt')
        post=posterior_validation(b,held,cfg);write(C/'reports/posterior_validation.json',post)
        moe=PredictiveMoE();target=sum(p.numel() for p in moe.parameters());width=min(range(16,257),key=lambda w:abs(sum(p.numel() for p in PredictiveMoE(1,w,12).parameters())-target));single=PredictiveMoE(1,width,12)
        write(C/'reports/capacity_control.json',dict(moe_parameters=target,single_parameters=sum(p.numel() for p in single.parameters()),single_width=width,relative_difference=abs(sum(p.numel() for p in single.parameters())-target)/target,same_total_gaussian_components=12,same_training_data_and_updates=True))
        train_model(moe,train,cfg['pilot_updates'],cfg,C,start,'pimoe_pilot');train_model(single,train,cfg['pilot_updates'],cfg,C,start,'single_control')
        m=fitted_evaluation(moe,held,b.sigma);s=fitted_evaluation(single,held,b.sigma)
        b.pimoe_work['scalar_conditional_likelihood_evaluations']+=2*sum(x['y'].numel()*len(x['belief'].theta) for x in held);write(C/'reports/predictive_moe_validation.json',m);write(C/'reports/single_control_validation.json',s)
        ms=m['summary'];ss=s['summary'];gates=dict(posterior=post['passes'],EIG_RMSE=ms['EIG_RMSE']<=.15,confident_ranking=ms['ranking_pairs']>=20 and ms['ranking_accuracy']>=.75,predictive_calibration=ms['KS']<=.1 and abs(ms['coverage90']-.9)<=.1,predictive_NLL=ms['NLL_excess']<=.1,not_worse_single=ms['EIG_RMSE']<=ss['EIG_RMSE']+.02 and ms['NLL']<=ss['NLL']+.05)
        write(C/'reports/preliminary_gate.json',dict(gates=gates,PASS=all(gates.values()),thresholds_frozen_before_training=True,no_official_seed1001_used=True))
        # Diagnostic internal immediate vs lookahead on the SAME held-out posterior.
        diagnostic=[]
        for item in held[:2]:
            for mode in [False,True]:
                planner=PIMoEPlanner(moe,b,seed=7004,particles=cfg['particles'],candidates=12,shortlist=2,fantasies=2,two_step=mode);planner.belief=item['belief'].fork(7005);a=planner.select();feat=state_features(b,planner.belief,a[None]);ref=direct_reference(feat['means'],feat['weights'],b.sigma,7006,cfg['reference_outer']);diagnostic.append(dict(two_step=mode,stage=item['stage'],selected_action=a.tolist(),selected_action_EIG=float(ref['EIG'][0]),reference_SE=float(ref['SE'][0]),selection_seconds=planner.timings[-1]));b.pimoe_work['scalar_conditional_likelihood_evaluations']+=len(planner.belief.theta)*cfg['reference_outer']
        write(C/'reports/lookahead_internal_ablation.json',diagnostic)
        if not all(gates.values()):status='PRELIMINARY_GATE_FAILED';why='No expensive full training or official EIG evaluation: predictive/posterior readiness failed';return
        train_model(moe,train,cfg['full_additional_updates'],cfg,C,start,'pimoe_final')
        finaldev=fitted_evaluation(moe,held,b.sigma);write(C/'reports/final_development_validation.json',finaldev);write(C/'reports/final_model_freeze.json',dict(checkpoint_sha256=sha(C/'models/pimoe_final.pt'),selection='last budget-eligible fixed-schedule checkpoint; no seed1001 selection',charged_seconds=time.monotonic()-start))
        status='TRAINING_COMPLETE_FINAL_EVALUATION_PENDING'
        if cfg.get('allow_final_evaluation',False):final_evaluate(moe,b,cfg,C,start);status='COMPLETE_SINGLE_SEED_PROTOTYPE'
    except Exception as exc:
        why=str(exc);write(C/'reports/failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:
        training_records=[json.loads(p.read_text()) for p in (C/'reports').glob('*_training.json')]
        cost=dict(optimizer_updates_by_component={p.stem:json.loads(p.read_text())['updates'] for p in (C/'reports').glob('*_training.json')},CPU_process_seconds=time.process_time(),charged_development_training_seconds=time.monotonic()-start,allowance_seconds=cfg['allowance_seconds'],actual_forward_parameter_stage_evaluations=b.calls,counts=b.pimoe_work,GPU_active_seconds=None,backward_FLOPs=None,simulator_backward_work='none: density learning does not differentiate the physical simulator',density_and_router_optimizer_counts='one joint update per predictive optimizer step; controls separately recorded',GPU_wall_note='Exclusive-device reserved wall time, not FLOPs or hardware-active time',ODE_work='not_applicable_location_algebraic_simulator',status=status,reason=why);write(C/'reports/resource_ledger.json',cost);write(C/'registry.json',dict(method='PIMoE-sBOED',seed=101,status=status,reason=why,final_EIG=None if not (C/'reports/final_evaluation.json').exists() else json.loads((C/'reports/final_evaluation.json').read_text())))
        print('FINAL_STATUS',json.dumps(cost),flush=True)
if __name__=='__main__':main()
