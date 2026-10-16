"""Separate Strategy-MoE trainer, preserving the original sparse MoE learner.

Same full-horizon likelihood, optimizer, episode budget and mild regularizers.
The new switching gate is clipped as its own disjoint parameter group.
"""
import torch
from src.research.sparse_moe import routing_gamma,stabilize_gradients

def train_family(b,h,s,seed,directory,variant,base_checkpoint=None,base_finetune=False,trajectory_budget=None,observer=None):
    import math,time,json
    from src.research.moe_architecture_controls import grad_norm
    from src.research.moe_family import MoESBOEDPolicy,component_groups,NAMES
    from src.research.engine import rollout
    torch.manual_seed(seed);start=time.perf_counter()
    budget=int(trajectory_budget or (s['updates']*s['batch_size']))
    revision=2
    if s['hidden']!=64:raise ValueError('Family experts must have widths64/64')
    if b.name not in {'location_finding','sir_ode'}:raise ValueError('Family benchmark not validated')
    if s.get('moe_gradient_clip_mode')!='component':raise ValueError('Family requires component clipping')
    p=MoESBOEDPolicy(variant,b,h,base_checkpoint=base_checkpoint,base_finetune=base_finetune,
        routing_mode=s.get('strategy_routing_mode','learned_switch'),
        initial_reroute_probability=s.get('strategy_initial_reroute_probability',.99),
        residual_scale=s.get('residual_scale',1.),residual_initial_std=s.get('residual_initial_std',.001))
    opt=torch.optim.Adam((v for v in p.parameters() if v.requires_grad),lr=s['learning_rate'],betas=(s.get('spce_adam_beta1',.8),s.get('spce_adam_beta2',.998)))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,s.get('spce_lr_decay',.98))
    records=[];count=0;step=0;contrasts=int(s['contrasts'])
    groups=component_groups(p);trace=directory/(variant+'_diagnostics.jsonl')
    while count<budget:
        n=min(s['batch_size'],budget-count);fraction=min(count/s.get('router_schedule_episodes',budget),1.)
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
            records.append(row);print(variant,{k:v for k,v in row.items() if not k.endswith('_by_stage')},flush=True)
        if observer is not None and observer.check(p,count,step,opt,scheduler,force=count>=budget):break
    journal=observer.finish(p) if observer is not None else {}
    if p.manager is not None:p.manager.reset()
    p.eval()
    from src.research.moe_family import measured_parameters
    measured=dict(first=measured_parameters(p,r['features'][0][:1],0),
        initial_routed=measured_parameters(p,r['features'][1][:1],1) if h>1 else None,
        later_routed=measured_parameters(p,r['features'][2][:1],2) if h>2 else None)
    if p.manager is not None:p.manager.reset()
    torch.save(p.checkpoint(),directory/(variant+'.pt'))
    base=getattr(p.core,'base',None)
    pre=base.provenance if base is not None else {}
    return p,dict(training='pathwise_full_horizon_sPCE',variant=variant,canonical_name=NAMES[variant],
        training_seconds=time.perf_counter()-start,measured_parameters=measured,generated_training_trajectories=count,
        contrast_training_trajectories=count*contrasts,training_trajectory_budget=budget,
        training_budget_unit='primary_episodes',history=records,diagnostic_trace=str(trace),
        base_checkpoint=pre,additional_training_episodes=count,
        total_training_episodes=count+pre.get('training_episodes',0),
        base_training_seconds=pre.get('training_seconds',0.),
        total_training_seconds=time.perf_counter()-start+pre.get('training_seconds',0.),
        policy_parameters=sum(v.numel() for v in p.parameters()),
        trainable_parameters=sum(v.numel() for v in p.parameters() if v.requires_grad),
        history_encoder_revision=2,encoder_input_dim=193,base_input_dim=pre.get('input_dim'),
        routing_warmup_fraction=s.get('moe_router_warmup_fraction',.3),
        routing_finish_fraction=s.get('moe_router_finish_fraction',.95),
        switching_schedule='none; preserves existing sigmoid gate',
        settings=p.config,checkpoint_selection=journal.pop('checkpoint_selection','final'),training_objective='full_horizon_sPCE',
        contrasts=contrasts,expert_ids=['E1','E2','E3','E4'],
        gradient_clip_mode='component',load_penalty_weight=1e-4,selector_entropy_weight_max=1e-4,**journal)
