"""Validated v1 loop with group-gradient logging only. No routing score term."""
from src.research.deepseek_sboed_experiment import *
def train_rc(C,cfg,variant,updates,budget,validator,credit=0.,start=None,policy_factory=None):
 out=C/'models'/variant;out.mkdir(parents=True,exist_ok=False);clock=ComputeClock(out/'compute_clock.json',budget,credit_seconds=credit,hardware=torch.cuda.get_device_name(0),start=start);torch.manual_seed(101)
 model=(LocationDAD(30,4.,3.,first_design_revision=1) if variant=='dad' else (policy_factory(variant=variant) if policy_factory is not None else DeepSeekPolicy(variant=variant))).cuda();model.train();
 if variant=='full' and cfg.get('charge_readiness_probe',True):
  probe=json.loads((C/'readiness/native_full_minibatch_probe.json').read_text());B0=probe['batch_size'];L0=probe['contrasts']
  for unit,n in [('fresh_complete_primary_trajectories',B0),('true_parameter_stages',B0*30),('contrast_parameter_stages',B0*L0*30),('scalar_conditional_likelihood_evaluations',B0*(L0+1)*30),('policy_optimizer_updates',1)]:clock.count(unit,n,component='readiness_probe_discarded_weights')
 b=make_benchmark(cfg['benchmark']);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']));scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,cfg['lr_decay']);records=[];evaluations=[];bound=1.;completed=0;stop='incomplete';unstable=collapsed=0
 try:
  for update in range(1,updates+1):
   if not clock.can_start(max(2.,bound*1.5)):stop='compute_budget_exhaustion';break
   unit=time.monotonic();charged_unit=clock.spent();opt.zero_grad()
   with torch.device('cuda'):
    if cfg.get('training_likelihood_backend') == 'streamed_checkpointed':
     from src.research.deepseek_sboed_v3 import streamed_spce
     r=streamed_spce(b,model,30,cfg['batch_size'],101+update*104729,cfg['contrasts'],cfg['contrast_chunk'])
    else:
     r=rollout(b,model,30,cfg['batch_size'],101+update*104729,cfg['contrasts'])
    info=r['info'][:,-1].mean();balance=model.balance_loss() if isinstance(model,DeepSeekPolicy) else info*0;loss=-info+balance
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-horizon information objective')
    loss.backward()
    if hasattr(model,'mask_unused_gradients'):model.mask_unused_gradients()
    gradient=norm(model.parameters())
    model._preclip_groups={k:norm(v) for k,v in model.groups().items()}
    if not math.isfinite(gradient):raise FloatingPointError('Nonfinite native gradient')
    if variant!='dad':torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    model._postclip_groups={k:norm(v) for k,v in model.groups().items()}
    opt.step();torch.cuda.synchronize()
   completed=update
   if not hasattr(model,'effective_expert_updates'):model.effective_expert_updates=[0]*7
   for e,expert in enumerate(model.routed):
    if any(p.grad is not None and bool((p.grad!=0).any()) for p in expert.parameters()):model.effective_expert_updates[e]+=1
   if (update-1)%1000==0:scheduler.step()
   B,L=cfg['batch_size'],cfg['contrasts'];clock.count('fresh_complete_primary_trajectories',B);clock.count('true_parameter_stages',B*30);clock.count('contrast_parameter_stages',B*L*30);clock.count('scalar_conditional_likelihood_evaluations',B*(L+1)*30);clock.count('policy_optimizer_updates',1)
   if 'work_counts' in r:
    for key,value in r['work_counts'].items():clock.count(key,value)
    clock.count('contrast_recomputation_parameter_stages',r['work_counts']['executed_parameter_stage_propagations']-B*(L+1)*30)
   if isinstance(model,DeepSeekPolicy):clock.count('shared_FFN_contexts',B*30 if model.shared is not None else 0);clock.count('routed_FFN_contexts',B*30*model.top_k)
   if update==1 or update%100==0:
    row=diagnostics(model,r,gradient,loss,info,update,clock.spent());row['learning_rate']=opt.param_groups[0]['lr'];row.update(effective_expert_updates=list(model.effective_expert_updates),gain=float(model.gain.detach()),preclip_group_norms=model._preclip_groups,postclip_group_norms=model._postclip_groups);records.append(row);print(variant,row,flush=True);write(C/'costs'/f'{variant}_progress.json',dict(records=records,clock=clock.report(),forward_parameter_stages=b.calls,counts=model.counts() if isinstance(model,DeepSeekPolicy) else dict(total_parameters=sum(p.numel() for p in model.parameters()))))
    unstable=unstable+1 if (gradient>1000 or row.get('normalized_late_context_RMS',0)>10) and update>=1000 else 0
    routing=row.get('routing',{});collapsed=collapsed+1 if routing and routing['distinct_selected_experts']<=model.top_k and update>=1000 else 0
    if unstable>=5:stop='persistent_gradient_instability';break
    if collapsed>=5:stop='persistent_routing_collapse';break
   if update%cfg['checkpoint_every']==0 or update==updates:
    path=out/f'update{update}.pt';save(model,path,update,cfg,opt,scheduler);clock.checkpoint(str(path))
   if variant=='full' and (update%cfg['development_every']==0 or update==cfg['control_updates']):
    with clock.exclude('validation'):
     values,meta=validator(model);np.save(C/'evaluations'/f'development_update{update}.npy',values);evaluations.append(dict(update=update,mean=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values))),metadata=meta));write(C/'reports/full_development_curve.json',evaluations)
    model.train()
   bound=max(bound,clock.spent()-charged_unit)
  else:stop='reference_schedule_completion'
  if completed==0:raise RuntimeError('No native optimizer update completed')
  # Final serialization is charged; eligibility is assessed after checkpoint IO.
  path=out/f'update{completed}_final.pt';save(model,path,completed,cfg,opt,scheduler);clock.checkpoint(str(path));selected=clock.selected();state=torch.load(selected['identifier'],map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);save(model,out/'selected.pt',state['update'],cfg)
  write(C/'costs'/f'{variant}_final.json',dict(stop_reason=stop,executed_updates=completed,selected_update=state['update'],checkpoint_sha256=sha(out/'selected.pt'),selected_checkpoint=selected,clock=clock.stop(),forward_parameter_stages=b.calls,CPU_process_seconds=time.process_time(),GPU_active_seconds=None,counts=model.counts() if isinstance(model,DeepSeekPolicy) else dict(total_parameters=sum(p.numel() for p in model.parameters())),reference_minibatch_preserved=True,planned_updates_this_phase=updates,cheap_control_only=variant!='full',training_reference_updates=50000,training_reference_episodes=100000000,legacy_episode_cap=None))
  return model.eval(),stop
 except Exception as exc:
  clock.stop();write(C/'reports'/f'{variant}_failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),updates=completed,clock=clock.report()));raise
