"""Fresh sPCE training and a documented variational actor-critic vsOED adaptation.

vsOED trains posterior, critic, and actor offline with replay and target networks.
Its actor gradients pass through Q, never through the simulator. gamma=1 retains
the total-information objective. Validation of all methods uses shared sPCE.
"""
import copy
import math
import time
import torch
from src.research.engine import rollout,generator
from src.research.policies import Posterior,mlp,make_policy,parameter_count


def training_contrasts(settings,fraction):
    """Predeclared schedule using training expenditure only, never evaluation."""
    maximum=int(settings['contrasts'])
    schedule=settings.get('spce_curriculum',1)
    if schedule==1:return maximum
    if schedule==2:return min(maximum,128 if fraction<.2 else 1024 if fraction<.5 else maximum)
    if schedule==3:return min(maximum,1024 if fraction<.2 else 4096 if fraction<.5 else maximum)
    raise ValueError('spce_curriculum must be 1, 2 or 3')


def select_checkpoint(benchmark,policy,horizon,settings,seed):
    with torch.no_grad():
        result = rollout(benchmark,policy,horizon,settings['validation_systems'],
                         seed,settings['validation_contrasts'])
        return float(result['info'][:,-1].mean())


def train_spce(benchmark,kind,horizon,settings,seed,directory,final_checkpoint=False,trajectory_budget=None,policy=None,observer=None):
    torch.manual_seed(seed)
    policy = policy if policy is not None else make_policy(benchmark,horizon,kind,settings['hidden'],settings)
    betas=(settings.get('spce_adam_beta1',.9),settings.get('spce_adam_beta2',.999))
    optimizer = torch.optim.Adam(policy.policy_parameters() if hasattr(policy,'policy_parameters') else policy.parameters(),lr=settings['learning_rate'],betas=betas)
    auxiliary=None
    if hasattr(policy,'posterior_view'):
        post=policy.posterior_view();qvalue=policy.critic_view()
        aux_params=list(post.parameters())+list(qvalue.parameters())
        if hasattr(policy,'information_experts'):aux_params+=list(policy.information_experts.parameters())+list(policy.information_router.parameters())
        auxiliary=torch.optim.Adam(aux_params,lr=settings.get('vsoed_posterior_lr',.001))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,settings.get('spce_lr_decay',1.))
    started,calls = time.perf_counter(),benchmark.calls
    physical_start = getattr(benchmark,'physical_calls',0)
    records,initializations = [],[]
    validation_seed = seed+10000019
    if kind=='fixed' and final_checkpoint:
        with torch.no_grad():
            if getattr(benchmark,'corrected_location',False):
                # Initialize from the source prior rather than the much wider box.
                locations=torch.randn(horizon,2,generator=generator(seed+817),dtype=torch.float64).clamp(-3.9,3.9)
                policy.sequence.copy_(locations if benchmark.design_bounds is None else torch.logit((locations+4)/8))
                label='fresh_source_prior_locations'
            else:
                policy.sequence.copy_(benchmark.random_latent(horizon,generator(seed+817)))
                label='fresh_random'
        initializations.append(dict(name=label,validation_spce=None))
    if kind == 'fixed' and not final_checkpoint:
        # Every candidate is feasible through the same sequential map.
        states = [('random',copy.deepcopy(policy.state_dict()))]
        if benchmark.name != 'location_finding':
            for label,q in [('short',.2),('central',.5),('long',.8)]:
                with torch.no_grad():
                    policy.sequence.fill_(math.log(q/(1-q)))
                states.append((label,copy.deepcopy(policy.state_dict())))
        else:
            with torch.no_grad():
                policy.sequence.copy_(torch.randn(policy.sequence.shape,dtype=torch.float64))
            states.append(('prior_sample',copy.deepcopy(policy.state_dict())))
        selected,best = None,-math.inf
        for label,state in states:
            policy.load_state_dict(state)
            value = select_checkpoint(benchmark,policy,horizon,settings,validation_seed)
            initializations.append(dict(name=label,validation_spce=value))
            if value is not None and value > best:
                selected,best = copy.deepcopy(state),value
        policy.load_state_dict(selected)
    best = select_checkpoint(benchmark,policy,horizon,settings,validation_seed) if not final_checkpoint else -math.inf
    best_state,best_update = copy.deepcopy(policy.state_dict()),0
    records.append(dict(update=0,validation_spce=None if final_checkpoint else best))
    primary_count=contrast_count=0
    updates=settings['updates']
    if trajectory_budget is not None:
        minimum_contrasts=training_contrasts(settings,0.)
        updates=math.ceil(trajectory_budget/settings['batch_size']) if settings.get('primary_episode_budget',0) else min(updates,math.ceil(trajectory_budget/((minimum_contrasts+1)*settings['batch_size'])))
    actual_update=0
    contrast_counts={}
    calibration=max(0,min(.5,settings.get('moe_router_calibration_fraction',0.)))
    for update in range(1,updates+1):
        fraction=(primary_count if settings.get('primary_episode_budget',0) else primary_count+contrast_count)/trajectory_budget if trajectory_budget is not None else (update-1)/updates
        contrast=training_contrasts(settings,fraction)
        batch=settings['batch_size']
        if trajectory_budget is not None:
            batch=min(batch,trajectory_budget-primary_count) if settings.get('primary_episode_budget',0) else min(batch,(trajectory_budget-primary_count-contrast_count)//(contrast+1))
        if batch<1:break
        refresh=None
        every=settings.get('moe_expert_refresh_every',0)
        if every and hasattr(policy,'select_expert') and policy.fixed_expert is None and update%every==0:
            available=policy.available_experts.nonzero().flatten().tolist()
            refresh=available[(update//every-1)%len(available)]
            policy.select_expert(refresh)
        result = rollout(benchmark,policy,horizon,batch,seed+update*104729,
                         contrast)
        primary_count+=batch
        contrast_count+=batch*contrast
        contrast_counts[contrast]=contrast_counts.get(contrast,0)+batch
        actual_update=update
        loss = -result['info'][:,-1].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite sPCE training loss')
        optimizer.zero_grad()
        loss.backward()
        if refresh is not None and hasattr(policy,'first_design'):
            policy.first_design.grad=None
        grad = torch.nn.utils.clip_grad_norm_(policy.policy_parameters() if hasattr(policy,'policy_parameters') else policy.parameters(),5.)
        if not torch.isfinite(grad):
            raise FloatingPointError('Nonfinite sPCE policy gradient')
        # Initially learn the gate against stable specialist proposals. Setting
        # gradients to None also prevents Adam momentum/weight decay updates.
        fraction=(primary_count if settings.get('primary_episode_budget',0) else primary_count+contrast_count)/trajectory_budget if trajectory_budget is not None else update/updates
        router_calibration=hasattr(policy,'branch_tokens') and policy.fixed_expert is None and fraction<=calibration
        if router_calibration:
            stable=list(policy.branch_tokens.parameters())+list(policy.experts.parameters() if hasattr(policy,'experts') else policy.network.parameters())+[policy.queries,policy.branch_empty]
            for parameter in stable:parameter.grad=None
        optimizer.step()
        posterior_loss=critic_loss=None
        if auxiliary is not None:
            f=torch.stack(result['features'],1).detach().reshape(-1,result['features'][0].shape[-1])
            next_f=torch.stack(result['features'][1:]+[result['final_features']],1).detach().reshape_as(f)
            z=benchmark.transform(result['truth']).detach()[:,None,:].expand(-1,horizon,-1).reshape(-1,benchmark.param_dim)
            posterior_loss=-post.log_prob(next_f,z).mean()
            before=torch.cat((torch.zeros(batch,1,device=result['info'].device,dtype=torch.float64),result['info'][:,:-1]),1)
            target=(result['info'][:,-1,None]-before).detach().reshape(-1,1)
            latent=torch.stack(result['latents'],1).detach().reshape(-1,benchmark.action_dim)
            critic_loss=(qvalue(torch.cat((f,latent),-1))-target).square().mean()
            information_loss=posterior_loss.new_zeros(())
            if hasattr(policy,'information_experts') and policy.fixed_expert is None:
                terminal=result['final_features'].detach()
                context=policy.context(terminal).detach()
                candidates=benchmark.transform(torch.cat((result['truth'][:,None],result['particles']),1)).detach()
                logits=torch.stack([expert(torch.cat((context[:,i,None].expand(-1,candidates.shape[1],-1),candidates),-1)).squeeze(-1) for i,expert in enumerate(policy.information_experts)],-1)
                weights=policy.role_weights(terminal,'information')
                scores=(logits*weights[:,None]).sum(-1)
                information_loss=-(scores[:,0]-torch.logsumexp(scores,-1)+math.log(candidates.shape[1])).mean()
            auxiliary.zero_grad();(posterior_loss+critic_loss+information_loss).backward()
            torch.nn.utils.clip_grad_norm_(aux_params,5.);auxiliary.step()
        if refresh is not None:policy.select_expert(None)
        if (update-1)%settings.get('spce_lr_schedule_every',1000)==0:scheduler.step()
        if update%settings['validate_every']==0 or update==updates:
            value = select_checkpoint(benchmark,policy,horizon,settings,validation_seed) if not final_checkpoint else None
            records.append(dict(update=update,train_spce=-float(loss.detach()),validation_spce=value,
                                posterior_nll=None if posterior_loss is None else float(posterior_loss.detach()),critic_mse=None if critic_loss is None else float(critic_loss.detach()),gradient_norm=float(grad),learning_rate=optimizer.param_groups[0]['lr'],
                                refreshed_expert=refresh,router_calibration=router_calibration,training_contrasts=contrast,training_spce_ceiling=math.log(contrast+1),
                                saturation_gap=math.log(contrast+1)+float(loss.detach())))
            label='train sPCE' if final_checkpoint else 'validation sPCE'
            display=-float(loss.detach()) if final_checkpoint else value
            print(f'{kind} update {update}: {label}={display:.6f}',flush=True)
            if value is not None and value > best:
                best,best_state,best_update = value,copy.deepcopy(policy.state_dict()),update
        if observer is not None and observer.check(policy,primary_count,update,optimizer,scheduler,force=update==updates):break
    if final_checkpoint:
        best_state,best_update=copy.deepcopy(policy.state_dict()),actual_update
        best=records[-1]['validation_spce']
    journal=observer.finish(policy) if observer is not None else {}
    if observer is not None:
        best_state=copy.deepcopy(policy.state_dict());best_update=journal['selected_checkpoint']['update'];best=journal['selected_checkpoint']['eig']
    policy.load_state_dict(best_state)
    torch.save(dict(kind=kind,architecture=getattr(policy,'architecture',type(policy).__name__),horizon=horizon,state_dict=best_state,
                    hidden=settings['hidden'],selected_update=best_update,
                    policy_settings={k:settings[k] for k in ['moe_revision','moe_router_temperature','spce_curriculum','first_design_revision'] if k in settings}),directory/f'{kind}.pt')
    return policy,dict(training='pathwise_sPCE',policy_architecture=getattr(policy,'architecture',type(policy).__name__),policy_parameters=parameter_count(policy),
                       selected_update=best_update,validation_spce=best,history=records,
                       initialization_candidates=initializations,training_seconds=time.perf_counter()-started,
                       actual_updates=actual_update,
                       generated_training_trajectories=primary_count,
                       contrast_training_trajectories=contrast_count,
                       training_forward_trajectories=primary_count+contrast_count,
                       training_trajectory_budget=trajectory_budget,training_budget_unit='primary_episodes' if settings.get('primary_episode_budget',0) else 'forward_including_contrasts',
                       expert_refresh_every=settings.get('moe_expert_refresh_every',0),primary_trajectories_by_contrast=contrast_counts,spce_curriculum=settings.get('spce_curriculum',1),
                       moe_revision=settings.get('moe_revision',1),moe_router_temperature=settings.get('moe_router_temperature',.5),
                       checkpoint_selection=journal.pop('checkpoint_selection','final' if final_checkpoint else 'best_validation'),
                       initial_learning_rate=settings['learning_rate'],lr_decay=settings.get('spce_lr_decay',1.),
                       adam_betas=list(betas),
                       lr_schedule_every=settings.get('spce_lr_schedule_every',1000),batch_size=settings['batch_size'],
                       forward_particle_stages=benchmark.calls-calls,
                       grid_particle_propagations_including_jacobians=getattr(benchmark,'physical_calls',0)-physical_start,**journal)


def train_vsoed(benchmark,horizon,settings,seed,directory,actor_kind="vsoed",trajectory_budget=None,actor=None,final_checkpoint=False):
    torch.manual_seed(seed)
    if actor is None and actor_kind=='vsoed':
        from src.research.policies import Policy
        actor=Policy(1+horizon*(benchmark.action_dim+benchmark.obs_dim+1),benchmark.action_dim,horizon,'vsoed',settings['hidden'])
    actor = actor if actor is not None else make_policy(benchmark,horizon,actor_kind,settings['hidden'],settings)
    inputs = 1+horizon*(benchmark.action_dim+benchmark.obs_dim+1)
    critic = mlp(inputs+benchmark.action_dim,settings['vsoed_hidden'],1).double()
    posterior = Posterior(inputs,benchmark.param_dim,settings['vsoed_hidden'],settings['vsoed_mixtures'])
    if hasattr(actor,'posterior_view'):
        posterior=actor.posterior_view();critic=actor.critic_view()
    target_actor=copy.deepcopy(actor)
    target_critic=target_actor.critic_view() if hasattr(actor,'critic_view') else copy.deepcopy(critic)
    actor_opt = torch.optim.Adam(actor.policy_parameters() if hasattr(actor,'policy_parameters') else actor.parameters(),lr=settings['learning_rate'])
    critic_opt = torch.optim.Adam(critic.parameters(),lr=settings['vsoed_critic_lr'])
    post_opt = torch.optim.Adam(posterior.parameters(),lr=settings['vsoed_posterior_lr'])
    rng = generator(seed+9109)
    replay=[]
    calls,started = benchmark.calls,time.perf_counter()
    physical_start = getattr(benchmark,'physical_calls',0)
    validation_seed = seed+10000019
    sample_only=actor_kind!='vsoed'
    generated=0
    best = select_checkpoint(benchmark,actor,horizon,settings,validation_seed) if not sample_only and not final_checkpoint else None
    best_state,best_aux,best_update = copy.deepcopy(actor.state_dict()),None,0
    records=[dict(update=0,validation_spce=best)]
    for update in range(1,settings['updates']+1):
        batch=settings['batch_size']
        if trajectory_budget is not None:batch=min(batch,trajectory_budget-generated)
        if batch<1:break
        exploration = settings['vsoed_exploration']*max(.1,1-update/settings['updates'])
        with torch.no_grad():
            result = rollout(benchmark,actor,horizon,batch,seed+update*104729,
                             0,exploration=exploration,track_likelihood=not sample_only)
            generated+=batch
            # Record raw latent actions, including exploration, as Q actions.
            feature_sequence = torch.stack(result['features']+[result['final_features']],1)
            latent_sequence = torch.stack(result['latents'],1)
            zs = benchmark.transform(result['truth'])
            for i in range(len(zs)):
                replay.append((feature_sequence[i].clone(),latent_sequence[i].clone(),zs[i].clone()))
            replay = replay[-settings['replay_capacity']:]
        selected = torch.randint(len(replay),(settings['batch_size'],),generator=rng)
        f = torch.stack([replay[int(i)][0] for i in selected])
        latent = torch.stack([replay[int(i)][1] for i in selected])
        z = torch.stack([replay[int(i)][2] for i in selected])
        # Fit all post-observation stages; incremental log-density rewards telescope.
        post_f = f[:,1:].reshape(-1,inputs)
        post_z = z[:,None].expand(-1,horizon,-1).reshape(-1,benchmark.param_dim)
        for _ in range(settings['vsoed_posterior_updates']):
            post_loss = -posterior.log_prob(post_f,post_z).mean()
            post_opt.zero_grad();post_loss.backward()
            torch.nn.utils.clip_grad_norm_(posterior.parameters(),5.)
            post_opt.step()
        with torch.no_grad():
            lp = posterior.log_prob(post_f,post_z).reshape(-1,horizon)
            before = torch.cat((benchmark.log_prior_z(z)[:,None],lp[:,:-1]),1)
            rewards = lp-before
            returns = rewards.flip(1).cumsum(1).flip(1)
            td = rewards.clone()
            for stage in range(horizon-1):
                next_action = target_actor(f[:,stage+1],stage+1)
                td[:,stage] += target_critic(torch.cat((f[:,stage+1],next_action),-1)).squeeze(-1)
            # As in vsOED TIG stabilization, gradually transition from MC to TD.
            factor = min(1.,update/max(1,settings['updates']//2))
            target = (1-factor)*returns+factor*td
        x = torch.cat((f[:,:-1],latent),-1).reshape(-1,inputs+benchmark.action_dim)
        for _ in range(settings['vsoed_critic_updates']):
            q = critic(x).reshape(-1,horizon)
            critic_loss = (q-target).square().mean()
            critic_opt.zero_grad();critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),5.);critic_opt.step()
        # Freeze critic weights while retaining its derivative with respect to action.
        for p in critic.parameters():p.requires_grad_(False)
        proposed = torch.stack([actor(f[:,stage],stage) for stage in range(horizon)],1)
        actor_loss = -critic(torch.cat((f[:,:-1],proposed),-1).reshape(-1,inputs+benchmark.action_dim)).mean()
        actor_opt.zero_grad();actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.policy_parameters() if hasattr(actor,'policy_parameters') else actor.parameters(),5.);actor_opt.step()
        for p in critic.parameters():p.requires_grad_(True)
        if not all(torch.isfinite(x) for x in [post_loss,critic_loss,actor_loss]):
            raise FloatingPointError('Nonfinite vsOED loss')
        with torch.no_grad():
            for model,target_model in ([(actor,target_actor)] if hasattr(actor,'critic_view') else [(actor,target_actor),(critic,target_critic)]):
                for p,q in zip(model.parameters(),target_model.parameters()):
                    q.lerp_(p,settings['target_rate'])
        if update%settings['validate_every']==0 or update==settings['updates']:
            value = select_checkpoint(benchmark,actor,horizon,settings,validation_seed) if not sample_only and not final_checkpoint else None
            records.append(dict(update=update,validation_spce=value,posterior_nll=float(post_loss.detach()),
                                critic_mse=float(critic_loss.detach()),actor_loss=float(actor_loss.detach()),
                                variational_terminal_reward=float(returns[:,0].mean()),exploration=exploration,
                                replay_size=len(replay)))
            print(f'{actor_kind} update {update}: '+(f'validation sPCE={value:.6f}' if value is not None else f'BA reward={float(returns[:,0].mean()):.6f}'),flush=True)
            if value is not None and value > best:
                best,best_state,best_update = value,copy.deepcopy(actor.state_dict()),update
                best_aux=dict(posterior=copy.deepcopy(posterior.state_dict()),critic=copy.deepcopy(critic.state_dict()))
    if sample_only or final_checkpoint:
        best_state=copy.deepcopy(actor.state_dict());best_update=update if generated else 0
        best_aux=dict(posterior=copy.deepcopy(posterior.state_dict()),critic=copy.deepcopy(critic.state_dict()))
    actor.load_state_dict(best_state)
    torch.save(dict(kind=actor_kind,architecture=getattr(actor,'architecture',type(actor).__name__),horizon=horizon,state_dict=best_state,hidden=settings['hidden'],
                    policy_settings=dict(moe_revision=4,moe_router_temperature=settings.get('moe_router_temperature',1.)),
                    selected_update=best_update,training_auxiliary=best_aux),directory/f'{actor_kind}.pt')
    return actor,dict(training='variational_BA_actor_critic_adaptation' if sample_only else 'variational_actor_critic_adaptation',gamma=1.,
                      likelihood_required=not sample_only,simulator_gradients_required=False,
                      checkpoint_selection='final' if sample_only or final_checkpoint else 'best_validation',
                      generated_training_trajectories=generated,contrast_training_trajectories=0,training_forward_trajectories=generated,
                      training_trajectory_budget=trajectory_budget,training_budget_unit='primary_episodes' if settings.get('primary_episode_budget',0) else 'forward_including_contrasts',policy_architecture=getattr(actor,'architecture',type(actor).__name__),
                      policy_parameters=parameter_count(actor),posterior_parameters=parameter_count(posterior),
                      critic_parameters=parameter_count(critic),selected_update=best_update,
                      validation_spce=best,history=records,training_seconds=time.perf_counter()-started,
                      forward_particle_stages=benchmark.calls-calls,
                      grid_particle_propagations_including_jacobians=getattr(benchmark,'physical_calls',0)-physical_start,
                      upstream='https://github.com/wgshen/vsOED/tree/d0884f9ba30fbddc56618841c68af2f3bfe16539',
                      adaptations=['shared ordered history and benchmark feasibility map',
                                   'joint diagonal-Gaussian mixture in transformed prior coordinates',
                                   'latent-action critic; incremental rewards; gamma=1',
                                   'final checkpoint for sample-only MoE; sPCE validation for explicit vsOED'])
