"""Frozen PIMoE validation only. No change to predictive architecture/loss/planner."""
import argparse,copy,csv,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import yaml
from src.research.pimoe import *
from src.research.pimoe_experiment import fitted_evaluation,train_model,write,sha
from src.research.benchmarks import make_benchmark

def read(p):return json.loads(Path(p).read_text())
def csvwrite(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def load_model(path,device='cpu'):
    x=torch.load(path,map_location='cpu',weights_only=False);m=PredictiveMoE(x['experts'],x['width'],x['components']);m.load_state_dict(x['state_dict'],strict=True);return m.to(device).eval()
def normalized_part(z,indices):
    w=z['weights'][:,indices];total=w.sum(-1,keepdim=True)
    if (total<1e-12).any():raise ValueError('Removal has negligible remaining predictive mass')
    return dict(weights=w/total,means=z['means'][:,indices],scales=z['scales'][:,indices])
def intervention(model,f,mode):
    if mode=='uniform_membership':
        handle=model.router.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
        try:return model(**f)
        finally:handle.remove()
    z=model(**f);K=model.components
    if mode=='full':return z
    if mode=='uniform_mixture_weights':
        z=dict(z);weights=z['weights'].reshape(len(z['weights']),4,K);weights=weights/weights.sum(-1,keepdim=True)/4;z['weights']=weights.flatten(1);return z
    if mode.startswith('expert_'):return normalized_part(z,list(range(int(mode[-1])*K,(int(mode[-1])+1)*K)))
    if mode.startswith('remove_'):return normalized_part(z,[i for i in range(4*K) if i//K!=int(mode[-1])])
    raise ValueError(mode)
def distribution_mean_variance(z):
    mu=(z['weights']*z['means']).sum(-1);var=(z['weights']*(z['scales'].square()+z['means'].square())).sum(-1)-mu.square();return mu,var

def js_distance(p,q,nodes=48):
    nodes,weights=np.polynomial.hermite.hermgauss(nodes);nodes=torch.as_tensor(nodes,dtype=DTYPE);weights=torch.as_tensor(weights/math.sqrt(math.pi),dtype=DTYPE)
    def expectation(z,other):
        A,K=z['weights'].shape;y=(z['means'][...,None]+math.sqrt(2)*z['scales'][...,None]*nodes).reshape(A,-1)
        a=PredictiveMoE.log_prob(z,y);b=PredictiveMoE.log_prob(other,y);v=a-torch.logaddexp(a,b)+math.log(2)
        return (v.reshape(A,K,-1)*weights*z['weights'][...,None]).sum((-1,-2))
    return .5*(expectation(p,q)+expectation(q,p))
def density_metrics(z,y,reference,reference_se,sigma):
    pred=(PredictiveMoE.entropy(z)-.5*math.log(2*math.pi*math.e*sigma**2)).cpu();lp=PredictiveMoE.log_prob(z,y);pit=PredictiveMoE.cdf(z,y).flatten().sort().values;uniform=(torch.arange(len(pit),dtype=DTYPE)+.5)/len(pit)
    delta=reference[:,None]-reference[None,:];mask=torch.triu(delta.abs()>2*(reference_se[:,None].square()+reference_se[None,:].square()).sqrt(),diagonal=1);agree=(pred[:,None]-pred[None,:]).sign()==delta.sign();n=int(mask.sum());hits=int(agree[mask].sum());chosen=int(pred.argmax())
    return dict(NLL=float(-lp.mean()),KS=float((pit-uniform).abs().max()),coverage90=float(((pit>.05)&(pit<.95)).double().mean()),EIG_RMSE=float((pred-reference).square().mean().sqrt()),EIG_bias=float((pred-reference).mean()),ranking_pairs=n,ranking_correct=hits,ranking_accuracy=hits/n if n else None,selected_reference_EIG=float(reference[chosen]),selected_reference_SE=float(reference_se[chosen]),selected_regret=float(reference.max()-reference[chosen]))

def new_diagnostic_states(model,b,C,episodes=4,particles=4096,candidates=12):
    g=rng(1006);states=[];histories=[]
    for episode in range(episodes):
        planner=PIMoEPlanner(model,b,seed=1006+episode*100003,particles=particles);truth=b.sample((1,),g);history=[]
        for stage in range(30):
            if stage in [0,4,9,14,19,24,29]:
                lo,hi=b.design_bounds;actions=lo+(hi-lo)*torch.rand(candidates,2,generator=g,dtype=DTYPE);f=state_features(b,planner.belief,actions)
                states.append(dict(episode=episode,stage=stage,features=f,belief=planner.belief.fork(1006+stage)))
            action=planner.select();mean=responses(b,truth,action[None])[0,0];y=mean+b.sigma*torch.randn((),generator=g,dtype=DTYPE);planner.observe(action,y);history.append(dict(action=action.tolist(),observation=float(y)))
        histories.append(dict(episode=episode,observed_history=history,belief_update_diagnostics=planner.belief.records));print('diagnostic_episode',episode+1,flush=True)
    torch.save(states,C/'evaluations/diagnostic_states_seed1006.pt');write(C/'reports/diagnostic_histories_seed1006.json',histories);return states

def diagnose(reference,C):
    cfg=yaml.safe_load((reference/'configs/pimoe_seed101.yaml').read_text());b=make_benchmark(cfg['benchmark']);model=load_model(reference/'models/pimoe_final.pt');before={k:v.clone() for k,v in model.state_dict().items()};started=time.monotonic();calls=b.calls
    statefile=C/'evaluations/diagnostic_states_seed1006.pt';states=torch.load(statefile,map_location='cpu',weights_only=False) if statefile.exists() else new_diagnostic_states(model,b,C)
    routing=[];distances=[];metrics=[];regional=[];pair_matrix=np.zeros((4,4));matcount=0;drawg=rng(1007);modes=['full']+['expert_'+str(i) for i in range(4)]+['remove_'+str(i) for i in range(4)]+['uniform_mixture_weights','uniform_membership'];saved={};expertmeans=[];expertvars=[];allpit=[];properties=[]
    with torch.no_grad():
        for index,item in enumerate(states):
            f=item['features'];z=model(**f);A,P=f['means'].shape;membership=z['membership'];mass=z['mass'];h=-(mass*mass.clamp_min(1e-300).log()).sum(-1);hard=torch.nn.functional.one_hot(membership.argmax(-1),4).double().mean(1);softH=-(membership*membership.clamp_min(1e-300).log()).sum(-1).mean(1)
            # Posterior response uncertainty, not label-sensitive theta covariance.
            weights=f['weights'];mu=(f['means']*weights).sum(-1);var=((f['means']-mu[:,None]).square()*weights).sum(-1);anchorvar=float(f['anchor_response'].var(0,unbiased=False).mean())
            for a in range(A):
                row=dict(state=index,episode=item['episode'],stage=item['stage'],candidate=a,assignment_entropy=float(h[a]),effective_experts=float(h[a].exp()),particle_assignment_entropy=float(softH[a]),collapse_max_alpha_over095=bool(mass[a].max()>.95),response_variance=float(var[a]),anchor_response_variance=anchorvar,action_x=float(f['actions'][a,0]),action_y=float(f['actions'][a,1]))
                for e in range(4):row['alpha_'+str(e)]=float(mass[a,e]);row['dominant_particle_frequency_'+str(e)]=float(hard[a,e])
                routing.append(row)
            # MC reference has independent observation randomness; density scores use fresh draws.
            ref=direct_reference(f['means'],weights,b.sigma,1007000+index,1024)
            ids=torch.multinomial(weights,512,True,generator=drawg);y=f['means'][:,ids]+b.sigma*torch.randn(A,512,generator=drawg,dtype=DTYPE)
            saved['membership_'+str(index)]=membership.numpy();saved['mass_'+str(index)]=mass.numpy();saved['actions_'+str(index)]=f['actions'].numpy();saved['reference_EIG_'+str(index)]=ref['EIG'].numpy();saved['reference_SE_'+str(index)]=ref['SE'].numpy()
            experts=[intervention(model,f,'expert_'+str(e)) for e in range(4)];em=[];ev=[]
            for e,z_e in enumerate(experts):
                mean,variance=distribution_mean_variance(z_e);em.append(mean.numpy());ev.append(variance.numpy())
                # Conditional regional calibration uses theta~p(theta|h)*r_e/alpha_e.
                conditional=membership[:,:,e]*weights[None];conditional/=conditional.sum(-1,keepdim=True);idx=torch.multinomial(conditional,512,True,generator=drawg);regiony=f['means'].gather(1,idx)+b.sigma*torch.randn(A,512,generator=drawg,dtype=DTYPE);pit=PredictiveMoE.cdf(z_e,regiony).flatten().sort().values;uniform=(torch.arange(len(pit),dtype=DTYPE)+.5)/len(pit)
                regional.append(dict(state=index,episode=item['episode'],stage=item['stage'],expert=e,region_NLL=float(-PredictiveMoE.log_prob(z_e,regiony).mean()),region_KS=float((pit-uniform).abs().max()),region_coverage90=float(((pit>.05)&(pit<.95)).double().mean())))
            expertmeans.append(np.stack(em,1));expertvars.append(np.stack(ev,1))
            for e in range(4):
                for j in range(e):
                    js=js_distance(experts[e],experts[j]);pair_matrix[e,j]+=float(js.mean());pair_matrix[j,e]+=float(js.mean());distances.append(dict(state=index,stage=item['stage'],expert_i=e,expert_j=j,mean_JS_nats=float(js.mean()),minimum_JS=float(js.min()),maximum_JS=float(js.max())))
            matcount+=1
            for mode in modes:
                zz=intervention(model,f,mode);m=density_metrics(zz,y,ref['EIG'],ref['SE'],b.sigma);metrics.append(dict(state=index,episode=item['episode'],stage=item['stage'],mode=mode,**m));saved[mode+'_EIG_'+str(index)]=(PredictiveMoE.entropy(zz)-conditional_entropy(b)).numpy()
            allpit.append(PredictiveMoE.cdf(z,y).numpy());print('diagnostic_state',index+1,flush=True)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items()),'Diagnostics mutated frozen model'
    np.savez_compressed(C/'evaluations/expert_arrays.npz',**saved,expert_predictive_means=np.concatenate(expertmeans),expert_predictive_variances=np.concatenate(expertvars),full_PIT=np.concatenate(allpit),pairwise_JS=pair_matrix/matcount)
    csvwrite(C/'reports/routing.csv',routing);csvwrite(C/'reports/expert_distribution_distances.csv',distances);csvwrite(C/'reports/contribution_diagnostics.csv',metrics);csvwrite(C/'reports/regional_calibration.csv',regional)
    summaries=[]
    for mode in modes:
        a=[x for x in metrics if x['mode']==mode];n=sum(x['ranking_pairs'] for x in a);summaries.append(dict(mode=mode,states=len(a),**{k:float(np.mean([x[k] for x in a])) for k in ['NLL','KS','coverage90','EIG_RMSE','EIG_bias','selected_reference_EIG','selected_regret']},ranking_accuracy=sum(x['ranking_correct'] for x in a)/n if n else None,ranking_pairs=n))
    csvwrite(C/'reports/contribution_summary.csv',summaries)
    from scipy.stats import spearmanr
    correlations=[]
    for e in range(4):
        for key in ['stage','response_variance','anchor_response_variance','action_x','action_y']:
            rho=spearmanr([x['alpha_'+str(e)] for x in routing],[x[key] for x in routing]).statistic;correlations.append(dict(expert=e,covariate=key,spearman_rho=float(rho),interpretation='Descriptive; dependent candidates/stages clustered within only4episodes; no p-value or semantic-role claim'))
    csvwrite(C/'reports/responsibility_correlations.csv',correlations)
    summary=dict(states=len(states),episodes=4,candidates_per_state=12,seed=1006,posterior_particles=4096,checkpoint_sha256=sha(reference/'models/pimoe_final.pt'),frozen_parameters_unchanged=True,mean_alpha=[float(np.mean([x['alpha_'+str(e)] for x in routing])) for e in range(4)],dominant_particle_frequency=[float(np.mean([x['dominant_particle_frequency_'+str(e)] for x in routing])) for e in range(4)],mean_effective_experts=float(np.mean([x['effective_experts'] for x in routing])),collapse_fraction=float(np.mean([x['collapse_max_alpha_over095'] for x in routing])),pairwise_JS_nats=(pair_matrix/matcount).tolist(),contributions=summaries,wall_seconds=time.monotonic()-started,physical_forward_stage_evaluations=b.calls-calls,limitations=['Outputs conditioned on approximate SMC beliefs; missing modes remain possible','Expert removal changes distribution; not causal retraining evidence','Expert-specific unconditional NLL is not the correct regional task; regional calibration separately reported','No per-expert sampling occurs; dominant frequencies are descriptive argmax summaries, not action-selection probabilities'])
    write(C/'reports/specialization_summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='contributions'},indent=2))

def train_control(reference,C,seed):
    from tools.journal_campaign import exclusive_device_lock
    lock=exclusive_device_lock();out=C/'location'/f'seed{seed}'/'single';(out/'models').mkdir(parents=True,exist_ok=True);(out/'reports').mkdir(exist_ok=True)
    if (out/'models/single_final.pt').exists():raise RuntimeError('Existing control preserved; no duplicate training')
    cfg=yaml.safe_load((reference/'configs/pimoe_seed101.yaml').read_text());cfg.update(device='cuda' if torch.cuda.is_available() else 'cpu',training_seed=seed)
    if seed!=101:raise ValueError('Initial validation gate only permits matched seed101 control')
    data=torch.load(reference/'models/posterior_training_states.pt',map_location='cpu',weights_only=False);model=load_model(reference/'models/single_control.pt');started=time.monotonic();begin_calls=data['training'][0]['belief'].b.calls
    updates=train_model(model,data['training'],cfg['full_additional_updates'],cfg,out,started,'single_final');assert updates==12000
    scores=fitted_evaluation(model,data['heldout'],.5);write(out/'reports/development.json',scores)
    old=read(reference/'reports/final_model_freeze.json');credit=old['charged_seconds'];elapsed=time.monotonic()-started
    pilot=read(reference/'reports/single_control_training.json');metadata=dict(model='matched_parameter_and_density_single_control',training_seed=seed,parameters=sum(p.numel() for p in model.parameters()),experts=1,components=12,width=96,pilot_updates=2000,followon_updates=updates,total_updates=2000+updates,phase_optimizer_reset_matches_original=True,training_data_sha256=sha(reference/'models/posterior_training_states.pt'),same_training_data_and_sample_order=True,new_training_wall_seconds=elapsed,original_pilot_wall_seconds=pilot['wall_seconds'],conservative_complete_credit_seconds=credit+elapsed,credit_note='Conservatively attribute entire original1210.63s development/data/pilots/follow-on credit plus new control training; not zero-cost shared data. Original phase-specific data time is unmeasured.',shared_physical_data_work=dict(physical_forward_evaluations_through_serialized_training_and_heldout_generation=begin_calls,simulator_count_scope='Shared dataset preparation, not repeated for control; posterior/reference auxiliary phases separate'),new_training_simulator_evaluations=0,checkpoint_sha256=sha(out/'models/single_final.pt'),hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',GPU_active_seconds=None,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,selection='predeclared last update; no official EIG selection',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(out/'reports/training_metadata.json',metadata);write(C/'readiness/single_seed101_trained.json',dict(status='PASS',**metadata));print(json.dumps(metadata,indent=2))


def train_replicate(reference,C,seed):
    from tools.journal_campaign import exclusive_device_lock
    if seed not in [102,103,104,105]:raise ValueError('Original seed101 is reused; replication seeds102-105 only')
    proof=C/'readiness/initial_comparison_ready.json'
    if not proof.exists() or read(proof).get('status')!='PASS':raise RuntimeError('Initial architecture/control comparison gate not passed')
    lock=exclusive_device_lock();out=C/'location'/f'seed{seed}'/'paired_training'
    if out.exists():raise RuntimeError('Existing replication artifacts preserved; no duplicate/restart')
    (out/'models').mkdir(parents=True);(out/'reports').mkdir()
    cfg=yaml.safe_load((reference/'configs/pimoe_seed101.yaml').read_text());cfg.update(device='cuda' if torch.cuda.is_available() else 'cpu',training_seed=seed)
    torch.manual_seed(seed);start=time.monotonic();data=torch.load(reference/'models/posterior_training_states.pt',map_location='cpu',weights_only=False)
    moe=PredictiveMoE();target=sum(p.numel() for p in moe.parameters());width=min(range(16,257),key=lambda w:abs(sum(p.numel() for p in PredictiveMoE(1,w,12).parameters())-target));single=PredictiveMoE(1,width,12)
    train_model(moe,data['training'],2000,cfg,out,start,'pimoe_pilot');train_model(single,data['training'],2000,cfg,out,start,'single_pilot')
    m=fitted_evaluation(moe,data['heldout'],.5);ss=fitted_evaluation(single,data['heldout'],.5);write(out/'reports/pilot_moe_validation.json',m);write(out/'reports/pilot_single_validation.json',ss)
    a=m['summary'];b=ss['summary'];gates=dict(EIG_RMSE=a['EIG_RMSE']<=.15,confident_ranking=a['ranking_pairs']>=20 and a['ranking_accuracy']>=.75,predictive_calibration=a['KS']<=.1 and abs(a['coverage90']-.9)<=.1,predictive_NLL=a['NLL_excess']<=.1,not_worse_single=a['EIG_RMSE']<=b['EIG_RMSE']+.02 and a['NLL']<=b['NLL']+.05,reference_posterior_validation=read(reference/'reports/posterior_validation.json')['passes'])
    write(out/'reports/readiness_gates.json',dict(gates=gates,PASS=all(gates.values()),same_frozen_original_thresholds=True))
    if not all(gates.values()):
        write(out/'reports/completion.json',dict(status='PRESERVED_SCIENTIFIC_READINESS_FAILURE',seed=seed,reason='Original readiness gate failed; no favorable-seed substitution',gates=gates));return
    train_model(moe,data['training'],12000,cfg,out,start,'pimoe_final');train_model(single,data['training'],12000,cfg,out,start,'single_final')
    write(out/'reports/final_moe_development.json',fitted_evaluation(moe,data['heldout'],.5));write(out/'reports/final_single_development.json',fitted_evaluation(single,data['heldout'],.5))
    elapsed=time.monotonic()-start;credit=read(reference/'reports/final_model_freeze.json')['charged_seconds'];counts=data['training'][0]['belief'].b.calls
    records={p.stem:read(p) for p in (out/'reports').glob('*_training.json')}
    result=dict(status='COMPLETE_TRAINING_EVALUATION_PENDING',seed=seed,model_optimizer_updates=14000,control_optimizer_updates=14000,training_data_sha256=sha(reference/'models/posterior_training_states.pt'),training_data_protocol='Original fixed training/heldout seeds1004/1005 and cached data shared across independent initialization/replay seeds; uncertainty conditional on this common training dataset',independent_model_initialization_and_replay_seed=seed,hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',new_paired_training_wall_seconds=elapsed,conservative_attributed_complete_credit_seconds=credit+elapsed,common_allowance_seconds=14400,unused_quota=max(0,14400-credit-elapsed),fresh_physical_simulations_this_replication=0,attributed_shared_dataset_forward_work=counts,shared_data_note='Physical dataset generated once; full conservative preparation/development credit and model work attributed to every method/seed',optimizer_updates_by_phase={k:v['updates'] for k,v in records.items()},wall_seconds_by_phase={k:v['wall_seconds'] for k,v in records.items()},peak_GPU_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,GPU_active_seconds=None,checkpoint_SHA256={name:sha(out/'models'/f'{name}_final.pt') for name in ['pimoe','single']},source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint_selection='Last fixed-schedule; no seed1001-based selection',original_pimoe_core_sha256=sha(Path(__file__).parent/'pimoe.py'))
    assert result['conservative_attributed_complete_credit_seconds']<14400
    write(out/'reports/completion.json',result);print(json.dumps(result,indent=2))


def equivalence_fixture(C):
    torch.manual_seed(101);b=make_benchmark(dict(system=dict(name='location_finding'),location_finding=dict(design_bounds=[-4,4]),observation=dict(noise_sigma=.5)));belief=AdaptiveBelief(b,256,1008);a=torch.tensor([[-2.,0.],[0.,0.],[2.,1.]],dtype=DTYPE);f=state_features(b,belief,a);model=PredictiveMoE().to(os.environ.get('PIMOE_FIXTURE_DEVICE','cpu'));ref=direct_reference(f['means'],f['weights'],.5,1009,128);loss,_,z=predictive_loss(model,f,ref['y'],ref['EIG'],ref['SE']);loss.backward();grad=torch.cat([p.grad.flatten() for p in model.parameters()]);opt=torch.optim.Adam(model.parameters(),lr=.001);opt.step()
    belief.update(torch.tensor([0.,0.],dtype=DTYPE),1.);belief.update(torch.tensor([1.,-1.],dtype=DTYPE),1.4);posterior_response=responses(b,belief.theta,a).detach().cpu().numpy()
    np.savez(C/('readiness/numerical_fixture_'+os.environ.get('PIMOE_FIXTURE_DEVICE','cpu')+'.npz'),posterior_response=posterior_response,posterior_log_weights=belief.logw.cpu().numpy(),weights=z['weights'].detach().cpu().numpy(),means=z['means'].detach().cpu().numpy(),scales=z['scales'].detach().cpu().numpy(),loss=np.asarray(float(loss.detach())),gradient=grad.cpu().numpy(),updated_parameters=torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy(),simulator_responses=f['means'].numpy());print('FIXTURE',sha(C/('readiness/numerical_fixture_'+os.environ.get('PIMOE_FIXTURE_DEVICE','cpu')+'.npz')))

def evaluate(reference,C,checkpoint,seed,model_label,model_loader=load_model):
    from tools.journal_campaign import exclusive_device_lock
    from src.research.sir_evaluation import contrast_scores
    lock=exclusive_device_lock();cfg=yaml.safe_load((reference/'configs/pimoe_seed101.yaml').read_text());b=make_benchmark(cfg['benchmark']);model=model_loader(checkpoint,'cuda' if torch.cuda.is_available() else 'cpu');canonical=Path(os.environ.get('PIMOE_CANONICAL_EVAL_DIR',cfg['canonical_evaluation_directory']))
    for name,key in [('location_cases.npz','cases_sha256'),('location_contrasts.npy','contrast_sha256')]:
        if sha(canonical/name)!=cfg[key]:raise ValueError('Canonical evaluation bank mismatch')
    out=C/'location'/f'seed{seed}'/model_label/'eval';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('Complete evaluation preserved; no duplicate')
    cases=np.load(canonical/'location_cases.npz');truth=torch.as_tensor(cases['truth'],dtype=DTYPE);noise=torch.as_tensor(cases['noise'],dtype=DTYPE);bank=torch.as_tensor(np.load(canonical/'location_contrasts.npy'),dtype=DTYPE);checkpoint_sha=sha(checkpoint);begin=time.monotonic();scores=[];latencies=[];totalstage=[]
    # Resume only finished trajectory rows, retaining exact original per-index RNGs.
    journal=out/'trajectory_records.jsonl'
    if journal.exists():
        for line in journal.open():
            row=json.loads(line);assert row['index']==len(scores) and row['checkpoint_sha256']==checkpoint_sha;scores.append(row['spce']);latencies+=row['decision_seconds'];totalstage+=row['complete_stage_seconds']
    for i in range(len(scores),2000):
        planner=PIMoEPlanner(model,b,seed=9000000+i,particles=cfg['planner_particles'],two_step=True);actions=[];ys=[];stage_times=[]
        for t in range(30):
            started=time.monotonic();a=planner.select();mu=responses(b,truth[i:i+1],a[None])[0,0];y=mu+b.sigma*noise[i,t,0];actions.append(a[None]);ys.append(y.reshape(1,1));planner.observe(a,y);stage_times.append(time.monotonic()-started)
        info,_=contrast_scores(b,dict(actions=actions,observations=ys,truth=truth[i:i+1]),bank,4096)
        row=dict(index=i,checkpoint_sha256=checkpoint_sha,spce=float(info[0,-1]),decision_seconds=planner.timings,complete_stage_seconds=stage_times,truth=truth[i].tolist(),measurement_noise_standard=noise[i].tolist())
        with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        scores.append(row['spce']);latencies+=planner.timings;totalstage+=stage_times
        if (i+1)%10==0:write(out/'progress.json',dict(trajectories=i+1,total=2000,elapsed_seconds=time.monotonic()-begin,checkpoint_sha256=checkpoint_sha));print(model_label,seed,'evaluation',i+1,flush=True)
    np.save(out/'scores.npy',scores);result=dict(model=model_label,training_seed=seed,EIG=float(np.mean(scores)),MC_SE=float(np.std(scores,ddof=1)/math.sqrt(2000)),systems=2000,contrasts=1000000,evaluation_seed=1001,checkpoint_sha256=checkpoint_sha,cases_sha256=cfg['cases_sha256'],contrast_sha256=cfg['contrast_sha256'],wall_seconds_this_session=time.monotonic()-begin,mean_selection_seconds=float(np.mean(latencies)),mean_complete_stage_seconds=float(np.mean(totalstage)),hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    if seed==101 and model_label=='single':
        old=np.load(reference/'evaluations/seed1001_scores.npy');d=old-np.asarray(scores);se=float(d.std(ddof=1)/math.sqrt(len(d)));result['PIMoE_minus_single']=dict(mean=float(d.mean()),paired_MC_SE=se,interval95=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],coupling='Identical canonical truth/noise/contrasts, planner initial and per-index RNGs; adaptive histories differ legitimately')
    write(out/'result.json',result);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pimoe-validation',action='store_true');p.add_argument('--action',choices=['diagnose','train-control','train-replicate','fixture','evaluate'],required=True);p.add_argument('--campaign',required=True);p.add_argument('--reference',required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--checkpoint');p.add_argument('--model-label',default='single');a=p.parse_args();C=Path(a.campaign).resolve();reference=Path(a.reference).resolve();torch.set_num_threads(2)
    # Algorithm implementation must be byte-identical to the frozen original.
    original=read(reference/'source_manifest.json')['sha256']['src/research/pimoe.py'];assert sha(Path(__file__).parent/'pimoe.py')==original
    if a.action=='diagnose':diagnose(reference,C)
    elif a.action=='train-control':train_control(reference,C,a.seed)
    elif a.action=='train-replicate':train_replicate(reference,C,a.seed)
    elif a.action=='fixture':equivalence_fixture(C)
    else:evaluate(reference,C,Path(a.checkpoint),a.seed,a.model_label)
if __name__=='__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: archived exploratory entry point')
if False:main()
