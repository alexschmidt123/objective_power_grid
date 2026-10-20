"""Pinned official ScoreBED training, with an explicit primary-episode cap."""
import os,sys,json,time,math,argparse,pathlib,subprocess
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('WANDB_MODE','disabled')
ROOT=pathlib.Path(__file__).resolve().parents[2]
VENDOR=ROOT/'src/research/vendor/scorebed'
sys.path.insert(0,str(VENDOR))
import numpy as np
import jax,jax.numpy as jnp
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from policy_learning.bed_models import LocationFinding
from policy_learning.loaders import load_score_trainer,load_policy_trainer
from policy_learning.eig_estimators import get_grad_estimate_fn
from policy_learning.utils.func import noop
from functools import partial
class CompatibleLocationFinding(LocationFinding):
    def __init__(self,*args,design_bounds=None,**kwargs):
        self.common_design_bounds=tuple(design_bounds) if design_bounds is not None else None
        super().__init__(*args,**kwargs)
    def feasible_design(self,x):
        if self.common_design_bounds is None:return x
        lo,hi=self.common_design_bounds
        return lo+(hi-lo)*jax.nn.sigmoid(x)
    def get_policy_output_activation(self,activation_type):
        if self.common_design_bounds is not None:return self.feasible_design
        return super().get_policy_output_activation(activation_type)
    def sample_designs(self,*args,**kwargs):
        x,aux,key=super().sample_designs(*args,**kwargs)
        return self.feasible_design(x),aux,key

    def _get_cached_fast_design_sampler(self,sampler_type,sampler_kwargs):
        # Upstream fast wrapper unpacked the dict into an API that expects
        # one sampler_kwargs argument. Keep the sampler itself unchanged.
        key=(sampler_type,self._freeze_sampler_kwargs(sampler_kwargs))
        if key not in self._fast_design_sampler_cache:
            self._fast_design_sampler_cache[key]=partial(self.sample_designs,sampler_type=sampler_type,sampler_kwargs=sampler_kwargs)
        return self._fast_design_sampler_cache[key]

COMMIT='cc8e996e3ee3237ab4c9bd3ef80e2e4b4aa9cee4'

def config(h,score_steps,batch):
    with initialize_config_dir(config_dir=str(VENDOR/'config'),version_base=None):
        cfg=compose(config_name='config',overrides=['experiment=location_finding'])
    cfg.bed_model.T=h
    cfg.score.num_iters=score_steps
    cfg.score.batch_size=batch
    cfg.policy.grad_est_type='score'
    cfg.policy.train_outer_samples=batch
    cfg.policy.train_outer_batch_size=batch
    cfg.n_restarts=1
    return OmegaConf.to_container(cfg,resolve=True)

def check_tree(tree,label):
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(tree)):
        raise FloatingPointError('Nonfinite official ScoreBED '+label)

def run(args):
    start=time.perf_counter();out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tiny=args.smoke;batch=4 if tiny else args.batch_size
    if batch<1:raise ValueError('Positive minibatch required')
    stats_count=4 if tiny else 10000
    restarts=args.restarts
    if restarts<1:raise ValueError('Positive restart count required')
    initialization=stats_count+4+10*restarts
    budget=initialization+16 if tiny else args.budget
    remaining=budget-initialization
    if remaining<2:raise ValueError('ScoreBED cap cannot cover initialization')
    journal_plan=json.loads(pathlib.Path(args.journal_plan).read_text()) if args.journal_plan else None
    score_count=8 if tiny else (min(int(journal_plan['score_pretraining_episodes']),remaining-1) if journal_plan else int(.7*remaining))
    policy_count=remaining-score_count
    if not tiny:
        from src.research.original_protocol import score_allocation
        allocation=score_allocation(budget,'location_finding',initialization,batch)
        if journal_plan and int(journal_plan['score_pretraining_episodes'])!=allocation['score']:raise ValueError('Frozen ScoreBED plan disagrees with published allocation')
        score_count,policy_count=allocation['score'],allocation['policy']
    cfg=config(args.horizon,int(journal_plan.get('score_schedule_reference_steps',math.ceil(score_count/batch))) if journal_plan else math.ceil(score_count/batch),batch)
    cfg['policy']['num_iters']=int(journal_plan.get('policy_schedule_reference_steps',math.ceil(policy_count/batch))) if journal_plan else math.ceil(policy_count/batch)
    cfg['bed_model']['init_seed']=args.seed
    cfg['common_design_bounds']=args.design_bounds
    cfg['n_restarts']=restarts
    if args.reference_sampler:
        cfg['score']['design_sampler_type']='time_corr_random_scale_gaussian'
        cfg['score']['design_sampler_kwargs']={'scale_min':.2,'scale_max':5.,'time_corr_rho_min':.7,'time_corr_rho_max':1.}
    (out/'scorebed_environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'scorebed_official_config.json').write_text(json.dumps(cfg,indent=2))
    # Use the official standardisation routine and charge its simulations.
    target=CompatibleLocationFinding(T=args.horizon,d=2,k=2,init_seed=args.seed,sigma=.5,standardise=False,design_bounds=args.design_bounds)
    target.compute_standardisation(stats_count)
    np.savez(out/'scorebed_normalization.npz',**{k:np.asarray(v) for k,v in target.get_std_stats_dict().items()})
    key=jax.random.key(args.seed)
    if args.reference_sampler:
        from policy_learning.utils.jax import derive_key
        key=derive_key(args.seed,'score_trainer')
    score,key=load_score_trainer(key,cfg,target,callbacks=[])
    from src.research.journal_scorebed_bridge import compute_clock_from_plan
    compute_clock=compute_clock_from_plan(journal_plan or {})
    native_unit_bound=0.
    score_spent=0;records=[]
    while score_spent<score_count:
        if compute_clock is not None and not compute_clock.can_start(native_unit_bound):
            from src.research.jax_checkpoint import save_state
            save_state(out/'score_pretraining_budget_exhausted.msgpack',score.training_state)
            raise RuntimeError('Whole-method compute budget exhausted during score pretraining: no trained policy available')
        unit_start=time.perf_counter()
        n=min(batch,score_count-score_spent)
        y,x,theta,aux,score.key=score._sample_target(score.key,(n,),use_fast_sampler=True)
        score.training_state,metrics=score.train_step(score.training_state,(y,x,theta,aux))
        # Synchronize for finite checks and honest wall timing.
        check_tree(metrics,'score metrics')
        score_spent+=n;step=int(score.training_state.step)
        native_unit_bound=max(native_unit_bound,time.perf_counter()-unit_start)
        if step%50==0 or score_spent==score_count:
            row=dict(phase='score',step=step,episodes=score_spent,metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1})
            records.append(row);print('ScoreBED official',row,flush=True)
            from src.research.resource_ledger import progress_ledger
            progress_ledger(out,'ScoreBED',initialization+score_spent,0,args.horizon,0,time.perf_counter()-start,actual_score_updates=step)
        if step%10000==0:
            from src.research.jax_checkpoint import save_state
            save_state(out/f'score_training_state_{step}.msgpack',score.training_state)
    check_tree(score.training_state.params,'score parameters')
    data_score,design_score=score.get_unbatched_score_fns()
    restart_records=[];restart_params=[];policy_spent=0;policy_updates=0
    for restart in range(restarts):
        if args.reference_sampler:key=derive_key(args.seed,'policy_trainer',restart)
        policy,key=load_policy_trainer(key,cfg,target,grad_fn=noop,callbacks=[])
        quota=policy_count//restarts+int(restart<policy_count%restarts)
        local_spent=0;last_n=None
        while local_spent<quota:
            if compute_clock is not None and not compute_clock.can_start(native_unit_bound):break
            unit_start=time.perf_counter()
            n=min(batch,quota-local_spent)
            if n!=last_n:
                grad_fn=get_grad_estimate_fn(policy.policy_network,target,data_score,design_score,N=n,outer_batch_size=n,M=1024,estimate_type='score')
                policy.update_grad_fn(grad_fn);last_n=n
            policy.training_state,metrics,_=policy.train_step(policy.training_state)
            check_tree(metrics,'policy metrics')
            policy_spent+=n;local_spent+=n;step=int(policy.training_state.step);policy_updates+=1
            native_unit_bound=max(native_unit_bound,time.perf_counter()-unit_start)
            if step%50==0 or local_spent==quota:
                row=dict(phase='policy',restart=restart,step=step,episodes=policy_spent,restart_episodes=local_spent,metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1})
                records.append(row);print('ScoreBED official',row,flush=True)
            from src.research.resource_ledger import progress_ledger
            progress_ledger(out,'ScoreBED',initialization+score_spent+policy_spent,0,args.horizon,policy_updates,time.perf_counter()-start,actual_score_updates=int(score.training_state.step),actual_policy_updates=policy_updates)
            if compute_clock is not None:
                from src.research.baseline_fidelity import scorebed_reported_params
                from src.research.journal_scorebed_bridge import maybe_validate
                if maybe_validate(scorebed_reported_params(policy.training_state,args.reference_sampler),args.journal_plan,initialization+score_spent+policy_spent,policy_updates,start):break
        if compute_clock is not None and policy_updates==0:raise RuntimeError('Compute quota produced no native policy update; do not report a trained ScoreBED policy')
        from src.research.baseline_fidelity import scorebed_reported_params
        params=scorebed_reported_params(policy.training_state,args.reference_sampler)
        restart_params.append(params)
        flat={f'{section}/{layer}/{name}':np.asarray(value) for section,leaves in params.items() for layer,values in leaves.items() for name,value in values.items()}
        np.savez(out/f'scorebed_restart{restart}_policy.npz',**flat)
        result=dict(restart=restart,primary_policy_episodes=quota,final_update=step)
        if args.journal_plan and restarts>1:
            # Native best-of-final-restarts reporting, selected on independent
            # diagnostic validation, never development/final evaluation draws.
            from src.research.baseline_fidelity import scorebed_restart_validation
            plan=dict(journal_plan);plan['validation']=scorebed_restart_validation(plan,tiny)
            folder=out/f'restart{restart}_validation';folder.mkdir(exist_ok=False)
            plan_path=folder/'plan.json';plan_path.write_text(json.dumps(plan,indent=2))
            from src.research.journal_scorebed_bridge import maybe_validate
            maybe_validate(params,str(plan_path),initialization+score_spent+policy_spent,step,start,force=True)
            state=json.loads((folder/'scorebed_journal_progress.json').read_text());result['validation_eig']=state['records'][-1]['eig']
        restart_records.append(result)
    chosen=max(range(restarts),key=lambda i:restart_records[i].get('validation_eig',-i))
    params=restart_params[chosen]
    if args.journal_plan:
        from src.research.journal_scorebed_bridge import maybe_validate,restored_params
        maybe_validate(params,args.journal_plan,initialization+score_spent+policy_spent,policy_updates,start,force=True)
        params=restored_params(params,args.journal_plan)
    check_tree(params,'policy parameters')
    arrays={}
    for section,leaves in params.items():
        for layer,values in leaves.items():
            for name,value in values.items():arrays[f'{section}/{layer}/{name}']=np.asarray(value)
    # Reference inputs/outputs verify the cross-framework history bridge.
    # Evaluation policies run on CPU in the shared evaluator. Generate the
    # bridge oracle on CPU as well, using the same trained weights. GPU TF32
    # and CPU float32 accumulation differ without changing the network map.
    cpu=jax.devices('cpu')[0]
    with jax.default_device(cpu):
        cpu_params=jax.device_put(params,cpu)
        key=jax.random.key(args.seed+10000019)
        x=jax.random.normal(key,(8,args.horizon,2));key,sub=jax.random.split(key)
        y=jax.random.normal(sub,(8,args.horizon,1));enc=jnp.ones((8,16))*.01
        expected=[]
        for t in range(args.horizon):
            expected.append(np.asarray(policy.policy_network.apply({'params':cpu_params},enc=enc,method='emitter')))
            enc=enc+policy.policy_network.apply({'params':cpu_params},y=y[:,t],x=x[:,t],method='encoder')
        arrays.update(reference_x=np.asarray(x),reference_y=np.asarray(y),reference_actions=np.stack(expected,1))
        if args.design_bounds is not None:arrays['common_design_bounds']=np.asarray(args.design_bounds)
    np.savez(out/'scorebed_official_policy.npz',**arrays)
    from flax import serialization
    (out/'scorebed_official_score.msgpack').write_bytes(serialization.to_bytes(score._score_variables_from_state()))
    data=dict(training='official_ScoreBED_JAX_score_matching_and_score_EIG_gradient',upstream_commit=COMMIT,
        generated_training_trajectories=initialization+score_spent+policy_spent,training_forward_trajectories=initialization+score_spent+policy_spent,
        contrast_training_trajectories=0,training_trajectory_budget=budget,training_budget_unit='primary_episodes',
        initialization_trajectories=initialization,standardisation_trajectories=stats_count,
        score_training_trajectories=score_spent,policy_training_trajectories=policy_spent,
        reported_policy_parameters='params_ema' if args.reference_sampler else 'params',
        policy_ema_rate=float(policy.training_state.ema_rate),
        random_streams='official derive_key: score_trainer, policy_trainer/restart' if args.reference_sampler else 'historical sequential split keys',
        restart_selection='best independent-validation EIG among final EMA policies' if restarts>1 else 'single final policy',
        budget_reconciliation=allocation if not tiny else dict(smoke=True),score_batch=batch,actual_score_updates=int(score.training_state.step),actual_policy_updates=policy_updates,policy_restarts=restart_records,selected_restart=chosen,reference_sampler=args.reference_sampler,
        training_seconds=time.perf_counter()-start,history=records,checkpoint_selection='final',
        architecture='official_4layer_256dim_8head_RFF_conservative_transformer',policy_architecture='official_DADNet_256_16_sum_encoder',
        adaptations='Declared native phase allocation including normalization and initialization; predeclared native final-policy restarts and paper AR1 sampler when enabled; capped score/policy allocation; independent validation restart selection; native batch1024, P1 location95:5 allocation; final partial minibatches; fast-sampler keyword-argument compatibility fix; no architecture or estimator substitutions',
        design_bounds=args.design_bounds,bounded_mapping='lo+(hi-lo)*sigmoid(raw_policy_output); teacher design sampler mapped into same box' if args.design_bounds else None,
        runtime=dict(jax=jax.__version__,devices=[str(d) for d in jax.devices()]))
    if not args.journal_plan and data['generated_training_trajectories']!=budget:raise AssertionError('Episode accounting mismatch')
    from src.research.resource_ledger import with_training_ledger
    data=with_training_ledger(data,args.horizon)
    (out/'scorebed_official_training.json').write_text(json.dumps(data,indent=2))
    print('SCOREBED_OFFICIAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--journal-plan');ap.add_argument('--budget',type=int,default=1000000);ap.add_argument('--horizon',type=int,default=30);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--batch-size',type=int,default=1024);ap.add_argument('--smoke',action='store_true');ap.add_argument('--design-bounds',type=float,nargs=2);ap.add_argument('--reference-sampler',action='store_true');ap.add_argument('--restarts',type=int,default=1);run(ap.parse_args())
