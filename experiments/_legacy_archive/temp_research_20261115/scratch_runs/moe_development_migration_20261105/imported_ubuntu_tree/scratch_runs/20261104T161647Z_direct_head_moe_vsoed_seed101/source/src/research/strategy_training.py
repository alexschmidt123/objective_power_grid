"""Separate Strategy-MoE trainer, preserving the original sparse MoE learner.

Same full-horizon likelihood, optimizer, episode budget and mild regularizers.
The new switching gate is clipped as its own disjoint parameter group.
"""
import torch
from src.research.sparse_moe import routing_gamma,stabilize_gradients

def train_strategy_moe(b,h,s,seed,directory,trajectory_budget=None):
    import math,time,json
    from src.research.moe_architecture_controls import grad_norm
    from src.research.strategy_moe import StrategyMoEPolicy,component_groups
    from src.research.engine import rollout
    torch.manual_seed(seed);start=time.perf_counter()
    budget=int(trajectory_budget or (s['updates']*s['batch_size']))
    revision=int(s.get('moe_history_encoder_revision',1))
    if revision==2 and b.name!='location_finding':raise ValueError('Sum correction is location-only')
    if revision!=2 or b.name!='location_finding':raise ValueError('Strategy pilot requires corrected location sum encoder')
    if s.get('moe_gradient_clip_mode')!='component':raise ValueError('Strategy training requires component clipping')
    p=StrategyMoEPolicy(h,s['hidden'],s.get('strategy_routing_mode','learned'),s.get('strategy_initial_reroute_probability',.99),action_scale=b.action_scale,obs_scale=b.obs_scale)
    opt=torch.optim.Adam(p.parameters(),lr=s['learning_rate'],betas=(s.get('spce_adam_beta1',.8),s.get('spce_adam_beta2',.998)))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,s.get('spce_lr_decay',.98))
    records=[];count=0;step=0;contrasts=int(s['contrasts'])
    groups=component_groups(p);trace=directory/'strategy_moe_sboed_diagnostics.jsonl'
    while count<budget:
        n=min(s['batch_size'],budget-count);fraction=count/budget
        # Smooth homotopy to binary DSelect routing; terminal phase uses sparse
        # selectors. The sPCE objective remains the information criterion.
        p.router.gamma=routing_gamma(fraction,s.get('moe_router_warmup_fraction',0.),s.get('moe_router_finish_fraction',.8))
        penalties=[];routes=[];strategies=[]
        def capture(module,inputs,output):
            penalties.append(module.regularizers())
            if module.last_route is not None:
                routes.append(module.last_route['weights'].detach())
                strategies.append({k:v.detach() for k,v in module.last_route['strategy'].items()})
        handle=p.register_forward_hook(capture)
        try:r=rollout(b,p,h,n,seed+(step+1)*104729,contrasts)
        finally:handle.remove()
        eig=r['info'][:,-1].mean()
        load=sum(v[0] for v in penalties)/h;entropy=sum(v[1] for v in penalties)/h
        loss=-eig+1e-4*load+1e-4*fraction*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sparse MoE loss')
        opt.zero_grad();loss.backward()
        before={k:grad_norm(v) for k,v in groups.items()}
        norm=stabilize_gradients(p,groups,s.get('moe_gradient_clip_mode','global'),5.)
        after={k:grad_norm(v) for k,v in groups.items()}
        opt.step();step+=1;count+=n
        if (step-1)%s.get('spce_lr_schedule_every',1000)==0:scheduler.step()
        row=dict(update=step,primary_episodes=count,training_eig=float(eig.detach()),selector_entropy=float(entropy.detach()),collapse_penalty=float(load.detach()),gamma=p.router.gamma,gradient_norm=float(norm),gradient_before=before,gradient_after=after,clip_scale=(min(1.,5./(float(norm)+1e-6)) if s.get('moe_gradient_clip_mode','global')=='global' else None),gradient_clip_mode=s.get('moe_gradient_clip_mode','global'),component_clip_scales={k:min(1.,5./(v+1e-6)) for k,v in before.items()})
        scores=r['info'][:,-1].detach()
        row['training_score_distribution']=dict(quantiles=torch.quantile(scores,torch.tensor([0.,.1,.5,.9,.99,1.],device=scores.device,dtype=scores.dtype)).cpu().tolist(),fraction_within_01_of_ceiling=float((scores>math.log(contrasts+1)-.1).double().mean()),fraction_within_05_of_ceiling=float((scores>math.log(contrasts+1)-.5).double().mean()))
        if step%s['validate_every']==0 or count==budget:row['training_scores']=scores.cpu().tolist()
        if routes:
            w=torch.stack(routes);choices=torch.nn.functional.one_hot(w.argmax(-1),4).to(w.dtype)
            row.update(router_probabilities_by_stage=w.mean(1).cpu().tolist(),hard_choice_frequency_by_stage=choices.mean(1).cpu().tolist(),execution_utilization_by_stage=(w>0).double().mean(1).cpu().tolist(),router_probabilities=w.mean((0,1)).cpu().tolist(),hard_choice_frequency=choices.mean((0,1)).cpu().tolist(),execution_utilization=(w>0).double().mean((0,1)).cpu().tolist(),routing_entropy=float((-(w*w.clamp_min(1e-15).log()).sum(-1)).mean()))
        if strategies and (step%int(s.get('strategy_diagnostics_every',1))==0 or count==budget):
            for key in ['reroute_probability','reroute_requested','actual_strategy_change','consecutive_duration']:
                values=torch.stack([v[key].double() for v in strategies])
                row['strategy_'+key+'_by_stage']=values.mean(1).cpu().tolist()
                row['strategy_'+key+'_mean']=float(values.mean())
            row['strategy_state_interpretation']='relaxed state and argmax proxies; actual hard behavior evaluated separately'
        with trace.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step%s['validate_every']==0 or count==budget:
            records.append(row);print('strategy_moe_sboed',{k:v for k,v in row.items() if not k.endswith('_by_stage')},flush=True)
    p.manager.reset();p.eval()
    torch.save(dict(kind='strategy_moe_sboed',architecture=p.architecture,horizon=h,hidden=s['hidden'],routing_k=1,gamma=p.router.gamma,state_dict=p.state_dict(),routing_mode=p.manager.mode,initial_reroute_probability=p.initial_reroute_probability,history_encoder_revision=revision,action_scale=b.action_scale,obs_scale=b.obs_scale),directory/'strategy_moe_sboed.pt')
    return p,dict(training='pathwise_full_horizon_sPCE_strategy_DSelect1',training_seconds=time.perf_counter()-start,generated_training_trajectories=count,contrast_training_trajectories=count*contrasts,training_forward_trajectories=count*(contrasts+1),training_trajectory_budget=budget,training_budget_unit='primary_episodes',history=records,policy_parameters=sum(v.numel() for v in p.parameters()),active_parameters_nonfirst=sum(v.numel() for n,v in p.named_parameters() if not n.startswith('experts.') and n!='first_design')+sum(v.numel() for v in p.experts[0].parameters()),history_encoder_revision=revision,effective_nonpadding_parameters=sum(v.numel() for v in p.parameters())-(4*s['hidden']+4)*(3*s['hidden']-16) if revision==2 else sum(v.numel() for v in p.parameters()),checkpoint_selection='final',diagnostic_trace=str(trace),routing_k=1,routing_warmup_fraction=s.get('moe_router_warmup_fraction',0.),routing_finish_fraction=s.get('moe_router_finish_fraction',.8),gradient_clip_mode=s.get('moe_gradient_clip_mode','global'),load_penalty_weight=1e-4,selector_entropy_weight_max=1e-4,hard_routing_at_evaluation=True,architecture=p.architecture,routing_mode=p.manager.mode,initial_reroute_probability=p.initial_reroute_probability,available_strategies=list(p.manager.available_strategies),strategy_diagnostics_every=int(s.get('strategy_diagnostics_every',1)))
