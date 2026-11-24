"""V1 training loop with only routed-expert contextual gradient adaptation."""
from contextlib import nullcontext
from src.research.deepseek_sboed_experiment import *
from src.research.context_expert_gradients import *

ORIGINAL_V1_TRAIN_FUNCTION_SHA256='f69a2e0799f30a99d074dcc56199f1097c20f026850e2123571cf17bc5c03179'

def save(model,path,update,cfg,opt=None,scheduler=None):
 state=dict(method='DeepSeekMoE-sBOED' if isinstance(model,DeepSeekPolicy) else 'corrected_dense_DAD_control',horizon=30,action_scale=4.,obs_scale=3.,variant=getattr(model,'variant','dad'),balance_factor=getattr(model,'balance_factor',0.),state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},training_seed=cfg['training_seed'],update=update,configuration=cfg,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 if opt is not None:state.update(optimizer=opt.state_dict(),scheduler=scheduler.state_dict())
 state['training_procedure']='context_gradient_specialization';state['architecture_source_commit']='13681d7091189a4e68056e6a2864dc58f9d5700b';torch.save(state,path)

def train_specialized(C,cfg,variant,updates,budget,validator,credit=0.,start=None):
 out=C/'models'/variant;out.mkdir(parents=True,exist_ok=False);clock=ComputeClock(out/'compute_clock.json',budget,credit_seconds=credit,hardware=torch.cuda.get_device_name(0),start=start);torch.manual_seed(cfg["training_seed"])
 model=(LocationDAD(30,4.,3.,first_design_revision=1) if variant=='dad' else DeepSeekPolicy(variant=variant)).cuda();model.train();
 if variant=='full':
  probe=json.loads((C/'readiness/native_full_minibatch_probe.json').read_text());B0=probe['batch_size'];L0=probe['contrasts']
  for unit,n in [('fresh_complete_primary_trajectories',B0),('true_parameter_stages',B0*30),('contrast_parameter_stages',B0*L0*30),('scalar_conditional_likelihood_evaluations',B0*(L0+1)*30),('policy_optimizer_updates',1)]:clock.count(unit,n,component='readiness_probe_discarded_weights')
 b=make_benchmark(cfg['benchmark']);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']));scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,cfg['lr_decay']);records=[];evaluations=[];expert_effective_updates=[0]*7;bound=1.;completed=0;stop='incomplete';unstable=collapsed=0
 try:
  for update in range(1,updates+1):
   if not clock.can_start(max(2.,bound*1.5)):stop='compute_budget_exhaustion';break
   unit=time.monotonic();opt.zero_grad()
   with torch.device('cuda'):
    special_step=cfg.get('specialization_enabled',True) and update%cfg['specialization_every']==0
    stage=cfg['specialization_stages'][(update//cfg['specialization_every']-1)%len(cfg['specialization_stages'])] if special_step else 1
    with (capture_context(model,stage) if special_step else nullcontext({})) as snapshot:
     r=rollout(b,model,30,cfg['batch_size'],cfg['training_seed']+update*104729,cfg['contrasts'])
    info=r['info'][:,-1].mean();balance=model.balance_loss();loss=-info+balance
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-horizon information objective')
    context=partition_context(snapshot) if special_step else None
    cg=None;special_seconds=0.;special_record=None
    if context is not None and context['valid']:
     torch.cuda.synchronize();special_begin=time.monotonic()
     cg=collect_context_gradients(model,r['info'][:,-1],balance,context)
     torch.cuda.synchronize();special_seconds=time.monotonic()-special_begin
    loss.backward()
    if cg is not None:
     special_begin=time.monotonic();special_record=apply_context_gradients(model,cg,context)
     torch.cuda.synchronize();special_seconds+=time.monotonic()-special_begin
     clock.count('specialization_context_backward_passes',2)
     clock.count('specialization_steps',1)
     clock.count('specialization_expert_adjustments',special_record['adjusted'])
     with (C/'costs/context_gradient_steps.jsonl').open('a') as f:f.write(json.dumps(dict(update=update,seconds=special_seconds,**special_record))+'\n')
    if special_step:clock.count('specialization_step_microseconds',round(special_seconds*1e6))
    gradient=norm(model.parameters())
    if not math.isfinite(gradient):raise FloatingPointError('Nonfinite native gradient')
    if variant!='dad':torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    opt.step();torch.cuda.synchronize()
   completed=update
   for e,expert in enumerate(model.routed):
    if any(p.grad is not None and bool((p.grad!=0).any()) for p in expert.parameters()):expert_effective_updates[e]+=1
   if (update-1)%1000==0:scheduler.step()
   B,L=cfg['batch_size'],cfg['contrasts'];clock.count('fresh_complete_primary_trajectories',B);clock.count('true_parameter_stages',B*30);clock.count('contrast_parameter_stages',B*L*30);clock.count('scalar_conditional_likelihood_evaluations',B*(L+1)*30);clock.count('policy_optimizer_updates',1)
   if isinstance(model,DeepSeekPolicy):clock.count('shared_FFN_contexts',B*30 if model.shared is not None else 0);clock.count('routed_FFN_contexts',B*30*model.top_k)
   if update==1 or update%100==0:
    row=diagnostics(model,r,gradient,loss,info,update,clock.spent());row['learning_rate']=opt.param_groups[0]['lr'];row.update(expert_effective_updates=list(expert_effective_updates),context_gradient=special_record,specialization_seconds_this_update=special_seconds,training_seed=cfg['training_seed']);records.append(row);print(variant,row,flush=True);write(C/'costs'/f'{variant}_progress.json',dict(records=records,clock=clock.report(),forward_parameter_stages=b.calls,counts=model.counts() if isinstance(model,DeepSeekPolicy) else dict(total_parameters=sum(p.numel() for p in model.parameters()))))
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
   bound=max(bound,time.monotonic()-unit)
  else:stop='reference_schedule_completion'
  if completed==0:raise RuntimeError('No native optimizer update completed')
  # Final serialization is charged; eligibility is assessed after checkpoint IO.
  path=out/f'update{completed}_final.pt';save(model,path,completed,cfg,opt,scheduler);clock.checkpoint(str(path));selected=clock.selected();state=torch.load(selected['identifier'],map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);save(model,out/'selected.pt',state['update'],cfg)
  write(C/'costs'/f'{variant}_final.json',dict(stop_reason=stop,executed_updates=completed,selected_update=state['update'],checkpoint_sha256=sha(out/'selected.pt'),selected_checkpoint=selected,clock=clock.stop(),forward_parameter_stages=b.calls,expert_effective_updates=expert_effective_updates,CPU_process_seconds=time.process_time(),GPU_active_seconds=None,counts=model.counts() if isinstance(model,DeepSeekPolicy) else dict(total_parameters=sum(p.numel() for p in model.parameters())),reference_minibatch_preserved=True,planned_updates_this_phase=updates,cheap_control_only=variant!='full',training_reference_updates=50000,training_reference_episodes=100000000,legacy_episode_cap=None))
  return model.eval(),stop
 except Exception as exc:
  clock.stop();write(C/'reports'/f'{variant}_failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),updates=completed,clock=clock.report()));raise
