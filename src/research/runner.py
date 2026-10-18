"""Root-launcher implementation of the EIG research plan (not a new launcher)."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from src.research.benchmarks import make_benchmark
from src.research.engine import generator,rollout,posterior_metrics,logmeanexp
from src.research.training import train_spce,train_vsoed

METHODS = ['random','fixed','dad','vsoed','rl_sboed','scorebed','moe_sboed']
STRATEGY_METHODS = ['strategy_moe_sboed']
FAMILY_METHODS = ['moe','resmoe','switchmoe','switchresmoe']
ABLATIONS = ['sum_single','attention_single','attention_dense_moe','attention_sparse_moe','attention_matched_dense','attention_matched_sparse','attention_uniform_moe','existing_moe_hard','existing_moe_relaxed','sum_single_193']
DEFAULTS = dict(strategy_routing_mode='learned',strategy_initial_reroute_probability=.99,strategy_diagnostics_every=1,moe_history_encoder_revision=1,first_design_revision=1,training_bound=1,variational_policy_lr=.0002,variational_density_lr=.001,
                variational_density_hidden=128,variational_mixtures=16,variational_density_updates=2,variational_warmup=200,smc_moves=8,moe_revision=4,moe_router_temperature=1.,spce_curriculum=1,updates=2000,batch_size=32,learning_rate=.001,hidden=64,contrasts=1024,
                validation_contrasts=1024,validation_systems=128,validate_every=100,
                eval_systems=128,eval_contrasts=1024,posterior_particles=1024,
                posterior_steps=64,posterior_chains=4,posterior_proposal_scale=.15,
                planner_particles=128,search_candidates=16,search_rounds=3,fantasies=32,
                vsoed_hidden=128,vsoed_mixtures=8,vsoed_critic_lr=.001,
                vsoed_posterior_lr=.001,vsoed_posterior_updates=3,vsoed_critic_updates=5,
                replay_capacity=8192,vsoed_exploration=.3,target_rate=.01,
                evaluation_batch_size=8,evaluation_contrast_chunk=16384,evaluation_population_systems=2000,threads=4,
                rl_batch_size=100,rl_replay_batch=4096,rl_replay_capacity=1000000,
                rl_min_buffer=100000,rl_gradient_steps=64,rl_log_every=10)


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class Tee:
    def __init__(self,stream,handle):self.stream,self.handle=stream,handle
    def write(self,value):
        self.handle.write(value);self.handle.flush()
        return self.stream.write(value)
    def flush(self):self.handle.flush();self.stream.flush()


def plot_results(output,all_rows,methods,seeds,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for seed in seeds:
        fig,ax=plt.subplots(figsize=(8,5))
        for method in methods:
            values=np.array([r['spce_by_stage'] for r in all_rows[(method,seed)]])
            x=np.arange(1,horizon+1)
            mean=values.mean(0)
            from src.research.moe_family import NAMES,ALIASES
            ax.plot(x,mean,label=NAMES.get(ALIASES.get(method,method),method))
            if len(values)>1:
                se=values.std(0,ddof=1)/math.sqrt(len(values))
                ax.fill_between(x,mean-1.96*se,mean+1.96*se,alpha=.1)
        ax.set(xlabel='Experiments collected',ylabel='Held-out EIG estimate (nats; sPCE)',
               title=f'One training seed; evaluation seed {seed}')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(output/'eval'/f'eig_curve_seed{seed}.png',dpi=160);plt.close(fig)


def freeze_source(root,output):
    """Scientific snapshot only: never copy local AGENTS.md or historical data."""
    paths = [p for p in (root/'src').rglob('*.py') if '__pycache__' not in p.parts]
    paths += [root/'run.sh',root/'sweep_run.sh']
    paths += list((root/'configs').glob('*.yaml'))
    paths += list((root/'src'/'research'/'vendor').rglob('LICENSE'))
    paths += list((root/'src'/'research'/'vendor').rglob('PROVENANCE.json'))
    paths += list((root/'src'/'research'/'vendor'/'scorebed'/'config').rglob('*.yaml'))
    hashes={}
    for source in paths:
        rel = source.relative_to(root)
        dest = output/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        hashes[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    write_json(output/'source_manifest.json',dict(git_commit=commit,sha256=hashes))


def evaluate(benchmark,policy,method,horizon,settings,seed,training,output,paper=False,*,infer_parameters=True,evaluation_draws=None):
    # Identity pairing is independent of training/decision RNG and batch partition.
    rng = generator(seed)
    population=settings.get('evaluation_population_systems',settings['eval_systems']) if paper else settings['eval_systems']
    if population<settings['eval_systems']:raise ValueError('Evaluation population smaller than requested subset')
    if evaluation_draws is None:
        truth = benchmark.sample((population,),rng)[:settings['eval_systems']]
        noise = torch.randn(population,horizon,benchmark.obs_dim,generator=rng,dtype=torch.float64)[:settings['eval_systems']]
        contrasts = benchmark.sample((settings['eval_contrasts'],) if paper or benchmark.name=='sir_ode' else
                                    (settings['eval_systems'],settings['eval_contrasts']),rng)
    streamed = paper or benchmark.name=='sir_ode'
    if evaluation_draws is not None:
        if not paper:raise ValueError('Explicit paired banks require the common streamed evaluator')
        truth,noise,contrasts=[evaluation_draws[k] for k in ('truth','noise','contrasts')]
        if truth.shape!=(settings['eval_systems'],benchmark.param_dim) or noise.shape!=(settings['eval_systems'],horizon,benchmark.obs_dim) or contrasts.shape!=(settings['eval_contrasts'],benchmark.param_dim):
            raise ValueError('Frozen evaluation cases/bank shape mismatch')
        if any(not torch.isfinite(x).all() for x in (truth,noise,contrasts)):raise ValueError('Nonfinite frozen evaluation inputs')
    if streamed:
        bank_path = output/'eval'/f'contrast_bank_seed{seed}.npy'
        if not bank_path.exists():np.save(bank_path,contrasts.numpy())
    started,calls = time.perf_counter(),benchmark.calls
    physical_start=getattr(benchmark,'physical_calls',0)
    rows=[]
    for start in range(0,len(truth),settings['evaluation_batch_size']):
        end=min(len(truth),start+settings['evaluation_batch_size'])
        with torch.no_grad():
            control_routes=[]
            def capture_control_route(module,inputs,output):
                route=getattr(module,'last_route',None)
                if route is not None:
                    w=route['weights'].detach()
                    control_routes.append(dict(weights=w.tolist(),proposals=[[] for _ in range(len(w))],
                        selected_expert=w.argmax(-1).tolist(),stage=inputs[1]))
                    if 'strategy' in route:
                        control_routes[-1]['strategy']={k:v.detach().tolist() for k,v in route['strategy'].items()}
            route_handle=policy.register_forward_hook(capture_control_route) if method in ABLATIONS or method in ['moe_sboed','strategy_moe_sboed']+FAMILY_METHODS else None
            try:
                result = rollout(benchmark,policy,horizon,end-start,seed+start*97451,
                                 0 if streamed else settings['eval_contrasts'],truth=truth[start:end],noise=noise[start:end],
                                 method=method,settings=settings,prior_particles=None if streamed else contrasts[start:end])
            finally:
                if route_handle is not None:route_handle.remove()
            if control_routes:result['routing']=control_routes
            # Denominator excluding truth: expected NMC upper estimator.
            if streamed:
                if benchmark.name=='sir_ode':
                    from src.research.sir_evaluation import contrast_scores
                else:
                    from src.research.location_evaluation import contrast_scores
                result['info'],upper = contrast_scores(benchmark,result,contrasts,settings['evaluation_contrast_chunk'])
            else:
                upper = result['log_likelihood'][:,0]-logmeanexp(result['log_likelihood'][:,1:])
            if infer_parameters:
                inference,posterior_artifacts = posterior_metrics(benchmark,result,settings,seed+start*97451+8000009,
                                                                 return_artifacts=True)
                np.savez_compressed(output/'eval'/f'{method}_seed{seed}_posterior_batch{start}.npz',
                                    **posterior_artifacts)
            else:inference=[None]*(end-start)
        if streamed and (start==0 or end%128==0 or end==len(truth)):
            print(f'{method} paper evaluation: {end}/{len(truth)} trajectories, L={settings["eval_contrasts"]}',flush=True)
        for i in range(end-start):
            rows.append(dict(method=method,eval_seed=seed,trajectory_id=start+i,
                             truth=truth[start+i].tolist(),measurement_noise_standard=noise[start+i].tolist(),
                             actions=[a[i].tolist() for a in result['actions']],
                             observations=[y[i].tolist() for y in result['observations']],
                             spce_by_stage=result['info'][i].tolist(),spce=float(result['info'][i,-1]),
                             nmc_upper_estimate=float(upper[i]),
                             inference=inference[i],
                             planning_diagnostics=[dict(stage=d['stage'],implementation=d['implementation'],
                                objective=d['objective'],truth_excluded=d['truth_excluded'],
                                **d['episodes'][i]) for d in result.get('planning_diagnostics',[])],
                             routing=[dict(weights=r['weights'][i],proposals=r['proposals'][i],stage=r.get('stage'),selected_expert=None if r.get('selected_expert') is None else r['selected_expert'][i],**({'strategy':{k:v[i] for k,v in r['strategy'].items()}} if 'strategy' in r else {})) for r in result['routing']],
                             mean_decision_seconds_per_trajectory=sum(result['decision_seconds'])/(horizon*(end-start))))
    scores=np.array([r['spce'] for r in rows])
    def inference_mean(key):
        return float(np.mean([r['inference'][key] for r in rows])) if infer_parameters else None
    summary=dict(method=method,eval_seed=seed,systems=len(rows),mean_spce=float(scores.mean()),
                 spce_mc_standard_error=float(scores.std(ddof=1)/math.sqrt(len(rows))) if len(rows)>1 else None,
                 mean_nmc_upper_estimate=float(np.mean([r['nmc_upper_estimate'] for r in rows])),
                 mean_mse=inference_mean('mse'),mean_normalized_mse=inference_mean('normalized_mse'),
                 mean_importance_ess=inference_mean('importance_ess'),
                 mean_decision_seconds=float(np.mean([r['mean_decision_seconds_per_trajectory'] for r in rows])),
                 evaluation_seconds=time.perf_counter()-started,forward_particle_stages=benchmark.calls-calls,
                 grid_particle_propagations_including_jacobians=getattr(benchmark,'physical_calls',0)-physical_start,
                 inference_degenerate_fraction=inference_mean('inference_degenerate'),
                 contrasts=settings['eval_contrasts'],spce_ceiling_nats=math.log(settings['eval_contrasts']+1),
                 eig_estimate_gap=float(np.mean([r['nmc_upper_estimate']-r['spce'] for r in rows])),
                 spce_near_ceiling_fraction=float(np.mean(scores>math.log(settings['eval_contrasts']+1)-.01)),
                 training_seconds=training.get('training_seconds',0.),
                 note='Monte Carlo estimator uncertainty within this training; not training-seed uncertainty')
    summary['parameter_inference_enabled']=bool(infer_parameters)
    if not infer_parameters:summary['mse_inference']='not_run_EIG_only; no MSE value is implied'
    elif getattr(benchmark,'corrected_location',False):
        summary['mse_inference']='two_independent_tempered_resample_move_SMC_populations'
        summary['ess_note']='Uniform weights after resampling; ESS is not a convergence certificate'
        summary['mean_replica_disagreement']=float(np.mean([r['inference']['independent_replica_mean_disagreement'] for r in rows]))
    return rows,summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='EIG research plan: common continuous benchmarks and offline design methods')
    parser.add_argument('--config',required=True)
    parser.add_argument('--T','--step-number',type=int)
    parser.add_argument('--variant',choices=FAMILY_METHODS)
    parser.add_argument('--training-episodes',type=int,help='Explicit additional primary-episode budget for a family timing/pilot run')
    parser.add_argument('--continuation-seconds',type=float,help='Elapsed training budget for explicit DAD-MLP continuation; stop after the first completed update reaching the target')
    parser.add_argument('--continuation-max-episodes',type=int,default=5000000,help='Safety cap for timed continuation, not its actual training budget')
    parser.add_argument('--dad-mlp-continue-from',help='Explicit pretrained base for matched additional-training sum_single control')
    parser.add_argument('--base-checkpoint',help='Explicit pretrained complete DAD-MLP base for residual training')
    parser.add_argument('--base-finetune',action='store_true',help='Separate controlled setting; default frozen base')
    parser.add_argument('--family-evaluate-from',help='Frozen family/legacy checkpoint; no training')
    parser.add_argument('--relaxed-routing',action='store_true',help='Diagnostic inference, not deployed hard policy')
    parser.add_argument('--eig-only',action='store_true',help='Keep common EIG evaluation; omit secondary posterior/MSE inference and record it as not run')
    parser.add_argument('--residual-scale',type=float,default=1.)
    parser.add_argument('--residual-initial-std',type=float,default=.001)
    parser.add_argument('--method','--methods',dest='methods')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--eval-seeds')
    parser.add_argument('--output')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--evaluate-from',help='Explicit completed upstream vsOED run; evaluation only')
    parser.add_argument('--native-evaluate-from',help='Evaluation-only native checkpoint; never trains a policy')
    parser.add_argument('--evaluation-relaxed-routing',action='store_true',help='Diagnostic relaxed family routing on the same saved evaluation bank')
    parser.add_argument('--evaluation-cases',help='Frozen NPZ truth/noise arrays for portable pairing')
    parser.add_argument('--evaluation-bank',help='Frozen NPY contrastive parameter bank')
    parser.add_argument('--objective',choices=['eig'],default='eig')
    parser.add_argument('--experiment_type',choices=['eig_based'],default='eig_based')
    parser.add_argument('--ablations',action='store_true')
    parser.add_argument('--source-check-manifest',help='Stop if a queued study scientific source has changed')
    parser.add_argument('--moe-diagnostic-from',help='Explicit original sparse checkpoint for hard/relaxed evaluation only')
    parser.add_argument('--noise_sigma','--noise-sigma',type=float)
    parser.add_argument('--window',type=float)
    parser.add_argument('--observation-kind',choices=['endpoint_rocof','max_absolute_rocof'])
    parser.add_argument('--N_obs','--N-obs',type=int)
    parser.add_argument('--min-duration-separation',type=float)
    for key,default in DEFAULTS.items():
        flags=['--'+key.replace('_','-')]
        if '_' in key:flags.append('--'+key)
        parser.add_argument(*flags,dest=key,type=type(default),default=None)
    parser.add_argument('--journal-phase',choices=['development','confirmation'],default='development')
    parser.add_argument('--journal',action='store_true',help='Enforce frozen aligned location publication protocol')
    parser.add_argument('--pilot',action='store_true',help='Enforce aligned settings for one train101/eval1001 pilot')
    args=parser.parse_args(argv)
    if args.pilot and args.journal:parser.error('Choose --pilot or --journal')
    root=Path(__file__).resolve().parents[2]
    if args.source_check_manifest:
        frozen=json.loads(Path(args.source_check_manifest).read_text())['sha256']
        changed=[rel for rel,sha in frozen.items() if not (root/rel).is_file() or hashlib.sha256((root/rel).read_bytes()).hexdigest()!=sha]
        if changed:parser.error('Queued study source changed; stopped rather than mixing implementations: '+','.join(changed[:10]))
    config_path=Path(args.config)
    if not config_path.exists():
        config_path=root/'configs'/(args.config.removesuffix('.yaml')+'.yaml')
    raw=yaml.safe_load(config_path.read_text())
    raw['_config_path']=str(config_path.resolve())
    raw.setdefault('research',{})
    if args.native_evaluate_from:
        from src.research.native_checkpoint_evaluation import main as evaluate_native
        return evaluate_native(args,raw,root)
    if args.journal and raw['research'].get('journal_protocol',{}).get('version')=='validation_convergence_v1':
        from src.research.journal_run import main as journal_main
        return journal_main(args,raw,root)
    paper = bool(raw['research'].get('paper_protocol'))
    journal_cfg=raw['research'].get('journal_protocol',{})
    default_train_seed=journal_cfg.get('training_seeds',[55101])[0] if args.journal and not args.smoke else 101
    args.seed = args.seed if args.seed is not None else default_train_seed
    default_eval_seeds=','.join(map(str,journal_cfg.get('evaluation_seeds',[551001]))) if args.journal and not args.smoke else '1001'
    args.eval_seeds = args.eval_seeds or default_eval_seeds
    if args.noise_sigma is not None:raw.setdefault('observation',{})['noise_sigma']=args.noise_sigma
    if args.window is not None:raw['research']['window']=args.window
    if args.observation_kind is not None:raw['research']['observation_kind']=args.observation_kind
    if args.N_obs not in {None,0}:parser.error('Research grid protocol uses one scalar RoCoF observation; N_obs=0')
    if args.min_duration_separation is not None:
        raw.setdefault('experiment',{})['min_duration_separation']=args.min_duration_separation
    horizon=args.T if args.T is not None else int(raw['experiment']['step_number'])
    methods=(args.methods.split(',') if args.methods else raw['experiment'].get('methods',METHODS))
    if args.training_episodes is not None:
        if not args.variant or args.training_episodes<1 or args.smoke or args.pilot or args.journal:
            parser.error('--training-episodes requires a positive non-smoke development family run')
        raw['research'].setdefault('paper_protocol',{})['nominal_training_trajectories']=args.training_episodes
        raw['research']['training_episode_cap']=args.training_episodes
    if args.variant:
        if args.methods:parser.error('Choose --variant or --methods')
        methods=[args.variant]
    if args.dad_mlp_continue_from and methods!=['sum_single']:parser.error('DAD-MLP continuation requires --method sum_single')
    if args.continuation_seconds is not None:
        if not args.dad_mlp_continue_from or methods!=['sum_single'] or args.smoke or args.pilot or args.journal or args.training_episodes is not None:
            parser.error('--continuation-seconds requires an explicit development sum_single continuation')
        if not math.isfinite(args.continuation_seconds) or args.continuation_seconds<=0 or args.continuation_max_episodes<1:
            parser.error('Timed continuation needs a finite positive duration and episode safety cap')
        raw['research'].setdefault('paper_protocol',{})['nominal_training_trajectories']=args.continuation_max_episodes
        raw['research']['training_episode_cap']=args.continuation_max_episodes
        raw['research']['continuation_time_budget_seconds']=args.continuation_seconds
        raw['research']['training_budget_unit']='additional_elapsed_seconds'
    if args.family_evaluate_from and not args.variant:parser.error('Frozen family evaluation requires --variant')
    if args.base_checkpoint and not args.variant:parser.error('Base checkpoint requires a residual --variant')
    if args.variant in {'resmoe','switchresmoe'} and not (args.base_checkpoint or args.family_evaluate_from):parser.error('Residual training requires --base-checkpoint')
    if args.base_finetune and args.variant not in {'resmoe','switchresmoe'}:parser.error('Base fine-tuning is residual-only')
    if args.ablations:methods=list(methods)+[m for m in ABLATIONS if m not in methods and not m.startswith('existing_')]
    if args.moe_diagnostic_from and (not set(methods)<= {'existing_moe_hard','existing_moe_relaxed'} or args.smoke):
        parser.error('--moe-diagnostic-from requires only existing_moe_hard/relaxed, non-smoke')
    if set(methods)& {'existing_moe_hard','existing_moe_relaxed'} and not args.moe_diagnostic_from:
        parser.error('Existing routing evaluations require explicit --moe-diagnostic-from')
    if set(methods)&set(ABLATIONS) and raw['system']['name']!='location_finding':
        parser.error('Architecture controls are location-only')
    if args.evaluate_from and (methods!=['vsoed'] or args.smoke or raw['system']['name']!='location_finding'):
        parser.error('--evaluate-from currently supports only non-smoke upstream location vsOED')
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=set(METHODS+ABLATIONS+STRATEGY_METHODS+FAMILY_METHODS):
        parser.error('Methods: '+','.join(METHODS+ABLATIONS+STRATEGY_METHODS+FAMILY_METHODS)+'; Myopic online optimization and Step-DAD are excluded; RL-sBOED currently supports location only')
    if 'scorebed' in methods and raw['system']['name'] not in {'location_finding','sir_ode'}:parser.error('ScoreBED supports location/SIR only')
    if 'rl_sboed' in methods and raw['system']['name'] not in {'location_finding','sir_ode'}:parser.error('RL-sBOED supports location/SIR only')
    settings=DEFAULTS|raw['research'].get('settings',{})
    for key in DEFAULTS:
        if getattr(args,key) is not None:settings[key]=getattr(args,key)
    if args.variant and args.ablations:parser.error('Family variants and legacy --ablations are separate experiments')
    if args.variant:
        # New-family defaults only: legacy commands and checkpoints retain their settings.
        family_defaults=dict(moe_history_encoder_revision=2,hidden=64,batch_size=64,learning_rate=5e-5,
            contrasts=1024,moe_gradient_clip_mode='component',moe_router_warmup_fraction=.3,
            moe_router_finish_fraction=.95,strategy_routing_mode='learned_switch')
        for key,value in family_defaults.items():
            if getattr(args,key,None) is None:settings[key]=value
        settings['residual_scale']=args.residual_scale
        settings['residual_initial_std']=args.residual_initial_std
    if args.smoke:
        settings.update(variational_warmup=1,variational_density_updates=1,updates=2,batch_size=4,contrasts=8,validation_contrasts=8,
                        validation_systems=4,validate_every=1,eval_systems=2,eval_contrasts=8,evaluation_population_systems=2,
                        posterior_particles=16,posterior_steps=4,posterior_chains=2,
                        planner_particles=8,search_candidates=4,search_rounds=1,fantasies=4,
                        vsoed_posterior_updates=1,vsoed_critic_updates=1,evaluation_batch_size=2,
                        rl_batch_size=4,rl_replay_batch=8,rl_replay_capacity=256,rl_min_buffer=4,rl_gradient_steps=2,rl_log_every=1)
        if raw['research'].get('paper_protocol',{}).get('corrected_methods'):
            settings['smc_moves']=2
    for key,value in settings.items():
        if key=='strategy_routing_mode':
            if value not in ('learned','learned_switch','always_reroute','always_continue'):parser.error('Invalid strategy mode')
            continue
        if key == "moe_gradient_clip_mode":
            if value not in ("global","component"):parser.error("Invalid MoE gradient clipping mode")
            continue
        if key=='residual_scale' and value==0:continue
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            parser.error(f'{key} must be finite and positive')
    if any(m in {'moe_sboed','uniform_moe','matched_dense'} for m in methods) and (settings['moe_revision']!=4 or settings['training_bound'] not in {1,3,4}):parser.error('Current MoE-sBOED is revision4; select sPCE(1) or sample-only BA actor-critic(3)')
    if settings['moe_history_encoder_revision'] not in (1,2):parser.error('MoE history encoder revision must be1 or2')
    if settings['moe_history_encoder_revision']==2 and raw['system']['name']!='location_finding' and not args.variant:parser.error('Sum-encoder correction is location-only')
    if settings['spce_curriculum'] not in {1,2,3}:parser.error('spce-curriculum must be 1, 2 or 3')
    if settings['training_bound'] not in {1,2,3,4}:parser.error('training-bound must be 1 (sPCE), 2 (pathwise BA) or 3 (sample-only BA actor-critic)')
    if settings['training_bound']==2 and (not paper or any(m not in {'moe_sboed','matched_dense','uniform_moe'} for m in methods)):
        parser.error('Variational development currently supports only location MoE and its architecture controls')
    if 'strategy_moe_sboed' in methods:
        if raw['system']['name']!='location_finding' or settings['moe_history_encoder_revision']!=2 or settings['training_bound']!=4 or settings.get('moe_gradient_clip_mode')!='component':
            parser.error('Strategy pilot requires location sum encoder, sPCE and component clipping')
        if not 0 < settings['strategy_initial_reroute_probability'] < 1:parser.error('Switch initialization must lie in (0,1)')
        if int(settings['strategy_diagnostics_every'])!=settings['strategy_diagnostics_every']:parser.error('Strategy diagnostic interval must be an integer')
    if horizon<1:parser.error('Horizon must be positive')
    if settings['target_rate']>1:parser.error('target-rate must be <=1')
    seeds=[int(s) for s in args.eval_seeds.split(',')]
    if len(set(seeds))!=len(seeds):parser.error('Evaluation seeds must be distinct')
    from src.research.comparison_protocol import validate_protocol
    try:
        protocol_record=validate_protocol(raw,settings,horizon,methods,args.seed,seeds,args.journal,args.smoke,args.pilot)
    except ValueError as exc:parser.error(str(exc))
    torch.set_num_threads(settings['threads'])
    now=datetime.now()
    default=root/'experiments'/now.strftime('%m%d%Y')/(now.strftime('%m%d%Y_%H%M%S_%f')+f"_{raw['system']['name']}_research_eig_T{horizon}_train{args.seed}")
    output=Path(args.output).resolve() if args.output else default
    # Explicit paths must remain within the required date-grouped hierarchy.
    rel=output.relative_to(root/'experiments')
    if len(rel.parts)<2 or len(rel.parts[0])!=8 or not rel.parts[0].isdigit() or not rel.parts[1].startswith(rel.parts[0]+'_'):
        parser.error('Output must be experiments/MMDDYYYY/MMDDYYYY_<campaign>/...')
    output.mkdir(parents=True,exist_ok=False)
    (output/'models').mkdir();(output/'eval').mkdir()
    (output/'logs').mkdir()
    log_handle=(output/'logs'/'run_log.log').open('w')
    sys.stdout=Tee(sys.stdout,log_handle);sys.stderr=Tee(sys.stderr,log_handle)
    from src.hardware import hardware_info
    raw['experiment'].update(methods=methods,step_number=horizon,experiment_type='eig_based')
    metadata=dict(protocol='moe_sboed_research_v1',settings=settings,benchmark=raw,hardware=hardware_info(),
                  train_seed=args.seed,eval_seeds=seeds,methods=methods,horizon=horizon,smoke=args.smoke,
                  evaluation_population_systems=settings['evaluation_population_systems'],evaluation_subset_rule='first N paired cases',
                  primary_metric='EIG (nats), estimated by common held-out sPCE',secondary_metric='posterior parameter MSE',
                  mse_inference='truth-excluded importance initialization plus random-walk MH in prior coordinates',
                  location_estimation='posterior samples aligned to MAP particle; final permutation-invariant squared error',
                  history='ordered padded design-observation pairs and stage index; history only',
                  online_policy_optimization=False,online_design_optimization=False,
                  grid_protocol='fixed physical bus 1, amplitude .05 pu, full state carry, increasing durations',
                  gradient='Torch pathwise; grid uses existing numerical state/duration Jacobians',
                  limitations=['Smoke establishes workability, not superiority',
                               'EIG contrast sensitivity and posterior convergence required before paper claims',
                               'vsOED is a documented adaptation, not an exact reproduction'])
    if paper:
        metadata['algorithm_implementations']={'vsoed':'Pinned upstream GMM-I with shared action-space, budget, discount and history adaptations',
            'dad':'Published location set encoder; shared feasible action map; sPCE training adaptation',
            'moe_sboed':'complete_branch_moe_v1; four branches containing policy, critic and posterior heads',
            'fixed':'Fresh sequence in shared action space optimized with sPCE',
            'random':'Independent '+raw['location_finding']['random_design_distribution']+' locations; separate action RNG'}
        metadata['limitations'][-1]='Comparison extends paper methods; only vsOED training algorithm is upstream'
        metadata['algorithm_implementations']['scorebed']='Pinned official JAX/Flax conservative Transformer, official sampler and score-gradient policy training; explicit capped budget and verified Torch inference bridge'
        metadata['training_budget_unit']='primary_episodes' if settings.get('primary_episode_budget',0) else 'forward_including_contrasts'
        metadata['history']='Method-specific history encoders: upstream one-hot/padded actor, DAD sum encoder, ordered MoE history'
        if raw['research']['paper_protocol'].get('corrected_methods'):
            metadata['history']='DAD retains published sum history; MoE and its controls share masked attention history; upstream vsOED retains original history'
            metadata['mse_inference']='Truth-excluded tempered resample-move SMC; ESS and MH diagnostics do not certify convergence'
            metadata['algorithm_implementations']['moe_sboed']='method_network_moe_v2: deterministic soft routing; v2 specialists; fixed contrast training with periodic single-expert EIG refresh'
    if settings['moe_revision']==2:
        metadata.setdefault('algorithm_implementations',{})['moe_sboed']='Fresh shared action head plus four normalized residual experts; dense soft routing at fixed temperature'
        metadata['architecture_controls']='Uniform-routing and parameter-matched dense share the same encoder, context and action head'
    if settings['moe_revision']==3:
        metadata.setdefault('algorithm_implementations',{})['moe_sboed']='Location-specific smooth spatial history anchors plus learned expert corrections and dense routing'
        metadata['architecture_controls']='Uniform-routing and matched-dense controls share normalized context, stage and four history anchors'
        metadata['location_anchor_settings']=dict(response_temperature_in_normalized_units=.2,spatial_radius=.15,repulsion_strength=8.,anchor_count=4)
    if 'rl_sboed' in methods:
        metadata.setdefault('algorithm_implementations',{})['rl_sboed']='RL-sBOED-I: standalone two-Q SAC/REDQ location adapter; incremental PCE rewards; gamma1'
        metadata['limitations'].append('RL-sBOED uses shared normalization and contrast budget; not an exact upstream reproduction')
    if args.variant:
        from src.research.moe_family import NAMES
        metadata['canonical_variant']=args.variant
        metadata['canonical_method']=NAMES[args.variant]
        metadata['family']='MoE-sBOED'
        metadata['historical_aliases']=['Strategy-MoE','strategy_moe_sboed'] if args.variant=='switchmoe' else []
        metadata['family_configuration']=dict(base_checkpoint=args.base_checkpoint,base_finetune=args.base_finetune,
            residual_scale=args.residual_scale,routing_mode=settings['strategy_routing_mode'],
            relaxed_evaluation=args.relaxed_routing,history_input_dim=193,base_input_dim=772 if 'resmoe' in args.variant else None)
        metadata['evaluation_seed_role']='development; seed1001 is not untouched final test data'
    metadata['comparison_protocol']=protocol_record
    metadata['training_bound']={1:'sPCE with auxiliary density/value fitting for MoE',2:'pathwise BA',3:'sample-only BA actor-critic',4:'method-specialist network MoE'}[settings['training_bound']]
    if args.variant:metadata['training_bound']='full-horizon sPCE; no auxiliary inference or reward networks'
    metadata['spce_training_schedule']={1:'fixed contrasts',2:'128 until 20 percent; 1024 until 50 percent; configured maximum thereafter',3:'1024 until 20 percent; 4096 until 50 percent; configured maximum thereafter'}[settings['spce_curriculum']]
    if 'scorebed' in methods:
        metadata['training_bound']='Official ScoreBED score-matching and score EIG policy gradient; same feasible design space at training and evaluation; common held-out EIG evaluator'
    if 'moe_sboed' in methods:
        metadata.setdefault('algorithm_implementations',{})['moe_sboed']='New sPCE-only shared-history four-policy-expert DSelect1; relaxed training, hard sparse inference; no auxiliary networks'
        metadata['architecture_controls']='New sparse method pilot; dense/top2/parameter-matched controls remain pending'
        metadata['training_bound']='Method-specific training: sPCE DAD/Fixed/new MoE, BA actor-critic vsOED, score-gradient ScoreBED; common held-out EIG evaluator'
        metadata['sparse_routing_protocol']=dict(k=1,gamma_start=1.,gamma_end=.001,anneal_fraction=settings.get('moe_router_finish_fraction',.8),warmup_fraction=settings.get('moe_router_warmup_fraction',0.),schedule=('delayed_cosine' if settings.get('moe_router_warmup_fraction',0.) else 'legacy_quadratic'),gradient_clip_mode=settings.get('moe_gradient_clip_mode','global'),collapse_weight=1e-4,selector_entropy_max_weight=1e-4,training_history_only=True)
        metadata['moe_history_encoder_revision']=settings['moe_history_encoder_revision']
        if settings['moe_history_encoder_revision']==2:
            metadata['history']='Permutation-invariant original location DAD sum16 encoder, fixed zero padding and unchanged stage coordinate; four existing64/64experts and DSelect1 router'
            metadata['algorithm_implementations']['moe_sboed']='sPCE-only sum-history four-policy-expert DSelect1 candidate; only the encoder changes'
    if 'strategy_moe_sboed' in methods:
        metadata.setdefault('algorithm_implementations',{})['strategy_moe_sboed']='Separate persistent strategy state; corrected sum encoder, existing four experts and DSelect1, learned sigmoid switching; full-horizon sPCE only'
        metadata['training_bound']='Full-horizon sPCE; unchanged mild selector/load penalties; no switch penalty or auxiliary networks'
        metadata['history']='Permutation-invariant corrected sum history plus stage; differentiable previous strategy distribution supplied to switching gate'
        metadata['strategy_protocol']=dict(mode=settings['strategy_routing_mode'],initial_reroute_probability=settings['strategy_initial_reroute_probability'],hard_switch_threshold=.5,switch_schedule='none',minimum_duration=None,available_strategies=['S0','S1','S2','S3'],training_state='q=(1-b)q_previous+b*p',evaluation_seed_role='1001 is development data',checkpoint_selection='final')
    metadata['parameter_inference_enabled']=not args.eig_only
    if args.eig_only:
        metadata['secondary_metric']=None
        metadata['mse_inference']='not_run_EIG_only; primary common EIG evaluator and all its settings are unchanged'
    write_json(output/'run_config.json',metadata)
    if set(methods)&set(ABLATIONS):
        metadata['architecture_control_protocol']=dict(controls=methods,encoder_comparison='Identical 772-input, 64/64 dense head; native DAD sum16 fixed-zero-padded versus full attention context',routing_algorithm='Existing DSelect1 unchanged',checkpoint_selection='final',simulation_cap=raw['research']['paper_protocol']['nominal_training_trajectories'],validation_seed=10000120,validation_eig_contrasts=65536,final_eig_contrasts=settings['eval_contrasts'])
        metadata['training_bound']='sPCE only; original sparse mild penalties retained; no posterior or RL critics'
        if methods==['sum_single_193']:
            metadata['architecture_control_protocol'].update(encoder_comparison='Exact corrected sum-MoE context: sum16, fixed padding, stage scalar;193-input64/64tanh single head',routing_algorithm='none_single_policy',validation_seed=None,validation_eig_contrasts=None,diagnostic_learning_curves=False,initialization='Same fresh untrained sum-MoE encoder/first action/expert0 at common seed; no checkpoint loading')
            metadata['algorithm_implementations']={'sum_single_193':'Fresh matched193-input single policy; existing corrected sum encoder and first-action treatment'}
            metadata['history']='Same permutation-invariant sum16 encoder and193-feature stage-augmented context as corrected MoE'
            metadata['training_bound']='Full-horizon sPCE; no router-specific penalties because control has no router; original MoE training unchanged'
        if args.moe_diagnostic_from:metadata['explicit_evaluation_source']=args.moe_diagnostic_from
        write_json(output/'run_config.json',metadata)
    print(f'RESULT_DIR={output}',flush=True)
    freeze_source(root,output)
    try:
        benchmark=make_benchmark(raw)
        summaries=[];all_rows={}
        for method in methods:
            if method in FAMILY_METHODS:
                from src.research.moe_family import load_policy,measured_parameters
                from src.research.family_training import train_family
                if args.family_evaluate_from:
                    policy=load_policy(args.family_evaluate_from,benchmark,mode=settings['strategy_routing_mode'],relaxed=args.relaxed_routing)
                    if policy.variant!=method or policy.horizon!=horizon:raise ValueError('Frozen evaluation variant/horizon mismatch')
                    training=dict(training='frozen_checkpoint_evaluation',training_seconds=0.,
                        source_checkpoint=str(Path(args.family_evaluate_from).resolve()),
                        source_checkpoint_sha256=hashlib.sha256(Path(args.family_evaluate_from).read_bytes()).hexdigest(),
                        not_an_independent_training=True)
                else:
                    with torch.device('cuda:0' if torch.cuda.is_available() and benchmark.name=='location_finding' else 'cpu'):
                        policy,training=train_family(benchmark,horizon,settings,args.seed,output/'models',method,
                            args.base_checkpoint,args.base_finetune,trajectory_budget=None if args.smoke else
                            raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',raw['research'].get('training_episode_cap',1000000)))
                    policy=policy.cpu().set_routing(settings['strategy_routing_mode'],args.relaxed_routing)
                training['canonical_variant']=method
            elif method=='random':
                policy,training=None,dict(training='none',training_seconds=0.)
            elif method in ABLATIONS or (method=='dad' and set(methods)&set(ABLATIONS)):
                from src.research.moe_architecture_controls import train_control,load_existing,EVALUATION_METHODS
                if method in EVALUATION_METHODS:
                    policy,training=load_existing(args.moe_diagnostic_from,method,horizon,raw)
                else:
                    train_settings=settings|raw['research'].get('training_profiles',{}).get('adaptive',{})
                    with torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'):
                        policy,training=train_control(benchmark,horizon,train_settings,args.seed,output/'models',method,trajectory_budget=None if args.smoke else raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',settings['updates']*settings['batch_size']),initial_checkpoint=args.dad_mlp_continue_from,time_budget_seconds=args.continuation_seconds)
                    policy=policy.cpu().eval()
            elif method=='strategy_moe_sboed':
                from src.research.strategy_training import train_strategy_moe
                train_settings=settings|raw['research'].get('training_profiles',{}).get('adaptive',{})
                with torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'):
                    policy,training=train_strategy_moe(benchmark,horizon,train_settings,args.seed,output/'models',trajectory_budget=None if args.smoke else raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',settings['updates']*settings['batch_size']))
                policy=policy.cpu().eval()
            elif method=='moe_sboed':
                from src.research.sparse_moe import train_sparse_moe
                train_settings=settings|raw['research'].get('training_profiles',{}).get('adaptive',{})
                with torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'):
                    policy,training=train_sparse_moe(benchmark,horizon,train_settings,args.seed,output/'models',
                        trajectory_budget=None if args.smoke else raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',settings['updates']*settings['batch_size']))
                policy=policy.cpu().eval()
            elif method=='scorebed':
                if benchmark.name=='sir_ode':
                    from src.research.sir_scorebed import train_scorebed
                else:
                    from src.research.scorebed import train_scorebed
                with torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'):
                    policy,training=train_scorebed(benchmark,horizon,settings,args.seed,output/'models',
                        trajectory_budget=None if args.smoke else raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories'))
                policy=policy.cpu()
            elif method=='rl_sboed':
                from src.research.research_redq import train_rl_sboed
                with torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'):
                    policy,training=train_rl_sboed(benchmark,horizon,settings,args.seed,output/'models',
                        trajectory_budget=None if args.smoke else raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories'))
                policy=policy.cpu()
            elif method=='vsoed':
                if args.evaluate_from:
                    from src.research.upstream_actor_evaluation import load_actor
                    policy,training=load_actor(args.evaluate_from,root,raw,horizon,args.seed)
                    metadata['explicit_evaluation_source']=training['source_run']
                    metadata['source_checkpoint_sha256']=training['source_checkpoint_sha256']
                    write_json(output/'run_config.json',metadata)
                elif raw['research'].get('upstream_vsoed'):
                    from src.research.upstream_vsoed import train_upstream_vsoed
                    profile=dict(raw['research']['upstream_vsoed'],_smoke=args.smoke,
                                 _trajectory_budget=raw['research']['paper_protocol']['nominal_training_trajectories'])
                    policy,training=train_upstream_vsoed(benchmark,horizon,settings,args.seed,output/'models',profile)
                else:
                    policy,training=train_vsoed(benchmark,horizon,settings,args.seed,output/'models',trajectory_budget=None if args.smoke else raw['research'].get('training_episode_cap'),final_checkpoint=True)
            else:
                train_settings=settings|raw['research'].get('training_profiles',{}).get(
                    'fixed' if method=='fixed' else 'adaptive',{})
                if paper and torch.cuda.is_available():
                    with torch.device('cuda:0'):
                        if settings['training_bound']==2:
                            from src.research.variational_pathwise import train_variational
                            policy,training=train_variational(benchmark,method,horizon,train_settings,args.seed,output/'models',
                                trajectory_budget=None if args.smoke else raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',raw['research'].get('training_episode_cap')))
                        else:
                            policy,training=train_spce(benchmark,method,horizon,train_settings,args.seed,output/'models',
                                final_checkpoint=True,trajectory_budget=None if args.smoke else
                                raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',raw['research'].get('training_episode_cap')))
                    policy=policy.cpu()
                elif settings['training_bound']==2:
                    from src.research.variational_pathwise import train_variational
                    policy,training=train_variational(benchmark,method,horizon,train_settings,args.seed,output/'models',
                        trajectory_budget=None if args.smoke else raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',raw['research'].get('training_episode_cap')))
                else:
                    policy,training=train_spce(benchmark,method,horizon,train_settings,args.seed,output/'models',
                        final_checkpoint=True,trajectory_budget=None if args.smoke else
                        raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',raw['research'].get('training_episode_cap')))
            checkpoint=output/'models'/f'{method}.pt'
            if checkpoint.exists():training['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if settings.get('primary_episode_budget',0) and not args.smoke and method not in {'random','existing_moe_hard','existing_moe_relaxed'} and not args.family_evaluate_from:
                required=raw['research'].get('paper_protocol',{}).get('nominal_training_trajectories',raw['research'].get('training_episode_cap'))
                if args.continuation_seconds is not None:
                    if not training.get('time_budget_reached'):raise ValueError('Timed control did not reach its declared training budget')
                    training['training_budget_unit']='additional_elapsed_seconds'
                elif training.get('generated_training_trajectories')!=required:
                    raise ValueError(f'{method}: expected {required} primary episodes, got '+str(training.get('generated_training_trajectories')))
                if args.continuation_seconds is None:training['training_budget_unit']='primary_episodes'
            write_json(output/'models'/f'{method}_training.json',training)
            for seed in seeds:
                rows,summary=evaluate(benchmark,policy,method,horizon,settings,seed,training,output,paper=paper,infer_parameters=not args.eig_only)
                write_json(output/'eval'/f'{method}_seed{seed}.json',rows)
                if method in FAMILY_METHODS and horizon>1:
                    from tools.report_moe_family import summarize_rows
                    write_json(output/'eval'/f'{method}_seed{seed}_strategy_management.json',summarize_rows(rows))
                summaries.append(summary);all_rows[(method,seed)]=rows
                mse_label='not run (EIG-only)' if args.eig_only else f"{summary['mean_normalized_mse']:.6f}"
                print(f"{method} seed {seed}: sPCE={summary['mean_spce']:.6f}; normalized MSE={mse_label}",flush=True)
        write_json(output/'summary.json',summaries)
        with (output/'summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        paired=[]
        plot_results(output,all_rows,methods,seeds,horizon)
        (output/'exit_code').write_text('0\n')
        print(f'RESULT_DIR={output}',flush=True)
    except Exception as exc:
        write_json(output/'failure.json',dict(exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        (output/'exit_code').write_text('1\n')
        raise
