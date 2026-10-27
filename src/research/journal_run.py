"""Validation-based journal path, entered exclusively through the root launcher."""
import copy,hashlib,json,math,sys,time,traceback
from datetime import datetime
from pathlib import Path
import torch
from src.research.journal_validation import StoppingRule,CommonValidation,ValidationObserver

METHODS=('random','fixed','dad','rl_sboed','vsoed','scorebed')

def main(args,raw,root):
    from src.research.runner import DEFAULTS,FAMILY_METHODS,write_json,freeze_source,evaluate,plot_results,Tee
    from src.research.benchmarks import make_benchmark
    from src.hardware import hardware_info
    protocol=raw['research']['journal_protocol'];phase=args.journal_phase
    if raw['system']['name'] not in {'location_finding','sir_ode'}:raise ValueError('Journal campaign excludes other benchmarks')
    if phase=='confirmation':
        raise ValueError('Confirmation is locked until development precision, native-method readiness and source/settings freeze are verified')
    seed=args.seed if args.seed is not None else protocol['development_training_seeds'][0]
    evaluation_seeds=[int(x) for x in (args.eval_seeds or '1001').split(',')]
    if 8831001 in evaluation_seeds:raise ValueError('Reserved final seed is forbidden during journal development')
    h=args.T or raw['experiment']['step_number'];methods=[args.variant] if args.variant else (args.methods.split(',') if args.methods else list(METHODS))
    if not set(methods)<=set(METHODS):raise ValueError('Unknown journal method')
    unsupported=set()

    if unsupported:raise NotImplementedError('Native validation adapters must pass readiness before journal training: '+','.join(sorted(unsupported)))
    settings=DEFAULTS|raw['research'].get('settings',{})
    requested_device=protocol.get('training_device','legacy_benchmark_default')
    train_device='cuda:0' if torch.cuda.is_available() and (raw['system']['name']=='location_finding' or requested_device=='cuda_if_available') else 'cpu'
    settings.update(raw['research']['journal_protocol'].get('common_pathwise_settings',{}))
    torch.set_num_threads(int(settings.get('threads',4)))
    validation_cfg=protocol['validation'];rule_cfg=protocol['stopping_rule']
    if args.smoke:
        validation_cfg=dict(validation_cfg,systems=4,contrasts=8,batch_size=2,chunk=4)
        rule_cfg=dict(rule_cfg,minimum_episodes=8,minimum_checkpoints=2,validation_every_episodes=4,patience=2,maximum_episodes=8,maximum_training_seconds=120)
        settings.update(batch_size=4,contrasts=8,validate_every=1,eval_systems=2,eval_contrasts=8,evaluation_population_systems=2,evaluation_batch_size=2,evaluation_contrast_chunk=4,updates=2)
    out=Path(args.output).resolve() if args.output else root/'experiments'/datetime.now().strftime('%m%d%Y')/(datetime.now().strftime('%m%d%Y_%H%M%S')+'_journal_'+raw['system']['name']+f'_T{h}_seed{seed}')
    rel=out.relative_to(root/'experiments')
    if len(rel.parts)<2 or len(rel.parts[0])!=8 or not rel.parts[1].startswith(rel.parts[0]+'_'):raise ValueError('Use dated experiment hierarchy')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['models','eval','logs']:(out/name).mkdir()
    handle=(out/'logs/run_log.log').open('w');sys.stdout=Tee(sys.stdout,handle);sys.stderr=Tee(sys.stderr,handle)
    raw=copy.deepcopy(raw);raw['experiment'].update(step_number=h,methods=methods)
    metadata=dict(protocol='journal_validation_convergence_v1',phase=phase,smoke=args.smoke,workability_only=args.smoke,train_seed=seed,eval_seeds=evaluation_seeds,horizon=h,methods=methods,settings=settings,benchmark=raw,hardware=hardware_info(),checkpoint_selection='final_predeclared_fixed_budget' if protocol.get('checkpoint_selection_mode')=='final_fixed_budget' else 'best_predeclared_validation_EIG',untouched_final_evaluation=False,parameter_inference_enabled=False)
    if args.evaluation_cases or args.evaluation_bank:
        if not args.evaluation_cases or not args.evaluation_bank:raise ValueError('Both paired evaluation files are required')
        metadata['paired_evaluation_files']={key:dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for key,path in [('cases',args.evaluation_cases),('contrasts',args.evaluation_bank)]}
    write_json(out/'run_config.json',metadata);freeze_source(root,out)
    compute_clock=None;credited_work=None;credited_updates=None
    compute_cfg=None if methods==['random'] else protocol.get('compute_budget')
    if compute_cfg:
        if len(methods)!=1:raise ValueError('One complete trained method per compute-budget job')
        import os
        from src.research.compute_budget import ComputeClock
        if not compute_cfg['hardware'].startswith('CPU_'):
            actual=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            if actual!=compute_cfg['hardware']:raise ValueError('Matched-GPU-time hardware mismatch: '+actual+' versus '+compute_cfg['hardware'])
        elif torch.cuda.is_available():raise ValueError('CPU accounting test must hide CUDA devices')
        credit=0.
        if methods[0] in {'resmoe','switchresmoe'}:
            if not args.base_checkpoint:raise ValueError('Residual base must be attributed before starting compute clock')
            base_dir=Path(args.base_checkpoint).resolve().parent
            records=list(base_dir.glob('*_training.json'))
            measured=[json.loads(p.read_text()).get('compute_budget') for p in records]
            measured=[p for p in measured if p is not None]
            if len(measured)!=1:raise ValueError('Exactly one measured base compute ledger required; episode estimates are not accepted')
            credit=measured[0]['charged_seconds']
            if measured[0]['hardware']!=compute_cfg['hardware']:raise ValueError('Residual base and residual phase use different hardware-time units')
            base_records=[json.loads(p.read_text()) for p in records]
            credited_work=next((p.get('resource_ledger',{}).get('complete_method_counts',p.get('resource_ledger',{}).get('counts')) for p in base_records if p.get('compute_budget') is not None),None)
            if credited_work is None:raise ValueError('Residual base model work is missing; time credit alone is insufficient')
            credited_updates=next((p.get('resource_ledger',{}).get('optimizer_updates') for p in base_records if p.get('compute_budget') is not None),None)
            if credited_updates is None:raise ValueError('Residual base optimizer-work counters missing')
        compute_clock=ComputeClock(out/'models/compute_clock.json',float(compute_cfg['seconds']),credit_seconds=credit,hardware=compute_cfg['hardware'],start=float(os.environ.get('BOED_TRAINING_START_MONOTONIC',time.monotonic())))
    try:
        validation_device=validation_cfg.get('device','cpu')
        if validation_device=='cuda_if_available':validation_device='cuda:0' if torch.cuda.is_available() else 'cpu'
        validator=CommonValidation(raw,h,validation_cfg['seed'],validation_cfg['systems'],validation_cfg['contrasts'],validation_cfg['batch_size'],validation_cfg['chunk'],device=validation_device)
        validator.preserve_progress_checkpoints=protocol.get('preserve_progress_checkpoints',False)
        write_json(out/'validation_draws.json',dict(seed=validator.seed,sha256=validator.identity,systems=validator.systems,contrasts=validator.contrasts))
        summaries=[];results={}
        for method in methods:
            b=make_benchmark(raw);s=settings|protocol.get('method_settings',{}).get(method,{})
            if args.smoke:s.update(rl_min_buffer=1,rl_batch_size=4,rl_replay_batch=8,rl_gradient_steps=1,rl_replay_capacity=64)
            this_rule=dict(rule_cfg)|protocol.get('method_stopping',{}).get(method,{})
            if args.smoke:this_rule=dict(rule_cfg)
            if args.smoke and method=='scorebed':this_rule['maximum_episodes']=24+10*int(s.get('scorebed_restarts',1))
            rule=StoppingRule(**this_rule);observer=ValidationObserver(validator,rule,out/'models',method,selection_mode='final_fixed_budget' if method=='random' else protocol.get('checkpoint_selection_mode','best_validation'),compute_clock=compute_clock,compute_checkpoint_seconds=900. if not compute_cfg else compute_cfg.get('checkpoint_seconds',900.))
            cap=rule.maximum_episodes
            reference_episodes=None
            if compute_cfg:
                from src.research.compute_budget import effective_episode_limit
                if 'reference_schedule_episodes' not in s:raise ValueError('Matched-compute mode requires an explicit reference schedule, separate from episode safety guard')
                reference_episodes=int(s['reference_schedule_episodes'])
                cap=effective_episode_limit(rule.maximum_episodes,reference_episodes)
                if method in {'dad','fixed'}:
                    from src.research.compute_budget import primary_reference_schedule
                    cap=primary_reference_schedule(s,rule.maximum_episodes)
            if method in {'resmoe','switchresmoe'} and not args.base_checkpoint:raise ValueError('Provide the same benchmark/seed-specific base checkpoint for residual variants')
            if method in {'resmoe','switchresmoe'} and not args.smoke:
                base_run=json.loads((Path(args.base_checkpoint).resolve().parent.parent/'run_config.json').read_text())
                if base_run.get('train_seed')!=seed:raise ValueError('Residual journal run requires its own independently trained seed-specific base')
            if method=='random':p=None;t=dict(training='none',training_seconds=0.,generated_training_trajectories=0,checkpoint_selection='not_applicable',convergence_status='not_applicable')
            elif method in FAMILY_METHODS:
                from src.research.family_training import train_family
                # The model/solver are device-independent; preserve CPU SIR default until GPU calibration passes.
                with torch.device(train_device):
                    p,t=train_family(b,h,s,seed,out/'models',method,args.base_checkpoint,args.base_finetune,cap,observer=observer)
            elif method=='sum_single':
                from src.research.moe_architecture_controls import train_control
                with torch.device(train_device):
                    p,t=train_control(b,h,s,seed,out/'models',method,cap,args.dad_mlp_continue_from,observer=observer)
            elif method=='scorebed':
                if b.name=='sir_ode':
                    from src.research.sir_scorebed import train_scorebed
                else:from src.research.scorebed import train_scorebed
                p,t=train_scorebed(b,h,s,seed,out/'models',cap,observer=observer)
            elif method=='vsoed':
                from src.research.upstream_vsoed import train_upstream_vsoed
                profile=dict(protocol.get('native_vsoed_profile',raw['research'].get('upstream_vsoed',{})))
                if not profile:raise ValueError('Missing native vsOED optimization profile')
                profile['updates']=max(1,(cap-profile['initial_replay'])//profile['new_trajectories'])
                profile['_trajectory_budget']=cap
                if args.smoke:profile.update(updates=2,new_trajectories=2,initial_replay=4,replay_batch=4,replay_capacity=32,posterior_updates=1,critic_updates=1,log_every=1)
                p,t=train_upstream_vsoed(b,h,s,seed,out/'models',profile,observer=observer)
            elif method=='rl_sboed':
                if s.get('rl_original_reference',False):
                    if b.name=='sir_ode':
                        from src.research.sir_native_rl import train_sir_native_rl as train_rl_sboed
                    else:
                        from src.research.native_rl_reference import train_native_rl as train_rl_sboed
                else:from src.research.research_redq import train_rl_sboed
                with torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'):
                    p,t=train_rl_sboed(b,h,s,seed,out/'models',cap,observer=observer)
                t['baseline_fidelity']='pinned_native_source_networks_and_update_kernels' if s.get('rl_original_reference',False) else 'documented_two_Q_Torch_adaptation_not_claimed_exact_published_reproduction'
            else:
                from src.research.training import train_spce
                with torch.device(train_device):
                    native=None
                    if method=='dad' and b.name=='sir_ode':
                        from src.research.sir_dad import SIRSumDAD
                        native=SIRSumDAD(h)
                    p,t=train_spce(b,method,h,s,seed,out/'models',final_checkpoint=True,trajectory_budget=cap,policy=native,observer=observer)
            if protocol.get('checkpoint_selection_mode')=='final_fixed_budget' and method!='random' and t.get('generated_training_trajectories')!=cap:
                raise RuntimeError('Fixed-budget training did not complete the exact episode allocation')
            if p is not None:p.cpu().eval()
            if 'optimization_seconds' in t:
                t['native_elapsed_seconds']=t['training_seconds'];t['training_seconds']=t['optimization_seconds'];t['total_offline_seconds']=t['training_seconds']+t['validation_seconds']
            t['journal_protocol']=protocol['version'];t['training_cap_is_not_target']=protocol.get('checkpoint_selection_mode')!='final_fixed_budget'
            if b.name=='sir_ode':
                t['ode_forward_steps']=None if method=='scorebed' else b.ode_forward_steps
                t['ode_forward_steps_scope']='actual adaptive RK4 primal steps in shared Torch simulator; gradient/replay work reported separately' if method!='scorebed' else 'JAX worker uses a separate solver; adaptive step counter unavailable, never reported as zero'
                t['minimum_RK4_steps_per_stage']=b.steps
            if 'resource_ledger' not in t and method!='random':
                from src.research.resource_ledger import with_training_ledger
                t=with_training_ledger(t,h)
            if compute_clock is not None:
                ledger=t['resource_ledger'];counts=ledger['counts']
                if credited_work is not None:
                    ledger['base_pretraining_counts']=credited_work
                    ledger['base_pretraining_optimizer_updates']=credited_updates
                    for component,n in credited_updates.items():
                        if n is not None:compute_clock.count('optimizer_updates',int(n),'base_'+component)
                    ledger['complete_method_counts']={k:None if counts.get(k) is None or credited_work.get(k) is None else counts[k]+credited_work[k] for k in set(counts)|set(credited_work)}
                    for unit,n in credited_work.items():
                        if n is not None:compute_clock.count(unit,n,'base_pretraining')
                else:ledger['complete_method_counts']=dict(counts)
                for unit,n in counts.items():
                    if n is not None:compute_clock.count(unit,n,'current_training')
                for component,n in ledger.get('optimizer_updates',{}).items():
                    if n is not None:compute_clock.count('optimizer_updates',int(n),component)
                import resource
                usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
                ledger['cpu_process_seconds']=usage.ru_utime+usage.ru_stime
                ledger['waited_child_cpu_seconds_including_validation']=children.ru_utime+children.ru_stime
                t['comparison_budget_unit']='exclusive_device_training_seconds';t['primary_episode_safety_cap']=cap
                t['compute_budget']=compute_clock.stop();ledger['compute_budget']=t['compute_budget']
                from src.research.compute_budget import classify_stop
                t['reference_schedule_episodes']=reference_episodes
                t['primary_episode_safety_cap']=rule.maximum_episodes
                t['stop_reason']=classify_stop(t.get('convergence_status'),t['generated_training_trajectories'],reference_episodes,rule.maximum_episodes)
                if t['generated_training_trajectories']>=reference_episodes:
                    t['stop_reason']='reference_schedule_completion'
                    if method=='scorebed' and t.get('actual_policy_updates',0)<s['policy_schedule_reference_steps']:t['stop_reason']='predeclared_compute_phase_schedule_completion'
                t['compute_budget']['overrun_seconds']=max(0.,t['compute_budget']['charged_seconds']-t['compute_budget']['budget_seconds'])
                if t['stop_reason']=='incomplete_or_interrupted':
                    write_json(out/'models'/f'{method}_training.json',t)
                    raise RuntimeError('Premature native training termination: neither declared schedule, compute allowance nor episode safety guard reached')
            t['effective_training_settings']=s
            t['resume_scope']='Completed phases reusable; exact mid-phase resume not supported for all native trainers; partial runs are preserved, not silently resumed'
            t['historical_runtime_overlap']='New runs are serialized; historical timings require separate provenance'
            write_json(out/'models'/f'{method}_effective_settings.json',s)
            checkpoint=out/'models'/f'{method}.pt'
            if checkpoint.exists():t['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            write_json(out/'models'/f'{method}_training.json',t)
            for es in evaluation_seeds:
                saved_draws=None
                if args.evaluation_cases or args.evaluation_bank:
                    if not args.evaluation_cases or not args.evaluation_bank or es!=1001:raise ValueError('Provide both frozen development arrays and seed1001')
                    import numpy as np
                    case_arrays=np.load(args.evaluation_cases)
                    saved_draws=dict(truth=torch.from_numpy(case_arrays['truth']).double(),noise=torch.from_numpy(case_arrays['noise']).double(),contrasts=torch.from_numpy(np.load(args.evaluation_bank)).double())
                rows,summary=evaluate(b,p,method,h,settings,es,t,out,paper=True,infer_parameters=False,evaluation_draws=saved_draws)
                summary.update(convergence_status=t['convergence_status'],phase=phase,workability_only=args.smoke)
                write_json(out/'eval'/f'{method}_seed{es}.json',rows);summaries.append(summary);results[(method,es)]=rows
        write_json(out/'summary.json',summaries);plot_results(out,results,methods,evaluation_seeds,h)
        (out/'exit_code').write_text('0\n');print('RESULT_DIR='+str(out),flush=True)
    except BaseException as exc:
        write_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));(out/'exit_code').write_text('1\n');raise
