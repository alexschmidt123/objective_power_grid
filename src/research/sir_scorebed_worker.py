"""Pinned official ScoreBED training, with an explicit primary-episode cap."""
import os,sys,json,time,math,argparse,pathlib,subprocess
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('WANDB_MODE','disabled')
ROOT=pathlib.Path(__file__).resolve().parents[2]
VENDOR=ROOT/'src/research/vendor/scorebed'
sys.path.insert(0,str(VENDOR))
sys.path.insert(0,str(ROOT))
from src.research.sir_scorebed_model import SIRScoreModel,SIRScorePolicyNet,SIRScoreGRUNet
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
        cfg=compose(config_name='config',overrides=['experiment=double_pend'])
    OmegaConf.update(cfg,"bed_model.T",h,force_add=True)
    cfg.score.num_iters=score_steps
    cfg.score.batch_size=batch
    cfg.policy.grad_est_type='score_chunked'
    cfg.policy.train_outer_samples=batch
    cfg.policy.train_outer_batch_size=min(256,batch)
    cfg.n_restarts=1
    return OmegaConf.to_container(cfg,resolve=True)

def check_tree(tree,label):
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(tree)):
        raise FloatingPointError('Nonfinite official ScoreBED '+label)

def run(args):
    start=time.perf_counter();out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tiny=args.smoke;batch=4 if tiny else args.batch_size
    if batch<1:raise ValueError('Positive minibatch required')
    stats_count=4 if tiny else 100000
    initialization=stats_count+4+10
    budget=34 if tiny else args.budget
    remaining=budget-initialization
    if remaining<2:raise ValueError('ScoreBED cap cannot cover initialization')
    journal_plan=json.loads(pathlib.Path(args.journal_plan).read_text()) if args.journal_plan else None
    score_count=8 if tiny else (min(int(journal_plan['score_pretraining_episodes']),remaining-1) if journal_plan else int(.7*remaining))
    policy_count=remaining-score_count
    if not tiny:
        from src.research.original_protocol import score_allocation
        allocation=score_allocation(budget,'sir_ode',initialization,batch)
        if journal_plan and int(journal_plan['score_pretraining_episodes'])!=allocation['score']:raise ValueError('Frozen ScoreBED plan disagrees with published allocation')
        score_count,policy_count=allocation['score'],allocation['policy']
    from src.research.sir_scorebed_compat import sir_score_config
    cfg=sir_score_config(config(args.horizon,int(journal_plan.get('score_schedule_reference_steps',math.ceil(score_count/batch))) if journal_plan else math.ceil(score_count/batch),batch))
    cfg['score']['design_sampler_type']='random_sampler'
    cfg['score']['design_sampler_kwargs']={}
    cfg['bed_model']['init_seed']=args.seed
    sir=json.loads(pathlib.Path(args.sir_config).read_text())
    cfg['bed_model']=dict(sir,name='sir_ode',T=args.horizon)
    (out/'scorebed_environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'scorebed_official_config.json').write_text(json.dumps(cfg,indent=2))
    # Use the official standardisation routine and charge its simulations.
    target=SIRScoreModel(T=args.horizon,init_seed=args.seed,**sir)
    target.compute_standardisation(stats_count)
    np.savez(out/'scorebed_normalization.npz',**{k:np.asarray(v) for k,v in target.get_std_stats_dict().items()})
    from policy_learning.utils.jax import derive_key
    key=derive_key(args.seed,'score_trainer')
    score,key=load_score_trainer(key,cfg,target,callbacks=[])
    from src.research.journal_scorebed_bridge import compute_clock_from_plan
    compute_clock=compute_clock_from_plan(journal_plan or {});native_unit_bound=0.
    score_spent=0;records=[]
    while score_spent<score_count:
        if compute_clock is not None and not compute_clock.can_start(native_unit_bound):
            from src.research.jax_checkpoint import save_state
            save_state(out/'score_pretraining_budget_exhausted.msgpack',score.training_state)
            raise RuntimeError('Whole-method compute quota exhausted before SIR score pretraining completed; no trained policy')
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
    check_tree(score.training_state.params,'score parameters')
    data_score,design_score=score.get_unbatched_score_fns()
    net=SIRScoreGRUNet(encoder_shapes=[256,256],encoding_dim=64,rnn_hidden_dim=64,rnn_layers=2,enc_act='relu',emit_act='relu',emitter_shapes=[256,256],shape=(args.horizon,1),min_std=1e-4,deterministic=False,time_scale=target.end,count_scale=target.population)
    key=derive_key(args.seed,'policy_trainer',0)
    policy,key=load_policy_trainer(key,cfg,target,grad_fn=noop,callbacks=[],policy_net=net)
    policy_spent=0;last_n=None
    while policy_spent<policy_count:
        if compute_clock is not None and not compute_clock.can_start(native_unit_bound):break
        unit_start=time.perf_counter()
        n=min(batch,policy_count-policy_spent)
        if n!=last_n:
            grad_fn=get_grad_estimate_fn(policy.policy_network,target,data_score,design_score,N=n,outer_batch_size=min(256,n),M=1024,estimate_type='score_chunked')
            policy.update_grad_fn(grad_fn);last_n=n
        policy.training_state,metrics,_=policy.train_step(policy.training_state)
        check_tree(metrics,'policy metrics')
        policy_spent+=n;step=int(policy.training_state.step)
        native_unit_bound=max(native_unit_bound,time.perf_counter()-unit_start)
        if args.journal_plan:
            from src.research.journal_scorebed_bridge import maybe_validate
            if maybe_validate(policy.training_state.params_ema,args.journal_plan,initialization+score_spent+policy_spent,step,start,force=policy_spent>=policy_count):break
        if step%50==0 or policy_spent==policy_count:
            row=dict(phase='policy',step=step,episodes=policy_spent,metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1})
            records.append(row);print('ScoreBED official',row,flush=True)
    if compute_clock is not None and policy_spent==0:raise RuntimeError('SIR ScoreBED produced no native policy update within compute quota')
    from src.research.baseline_fidelity import scorebed_reported_params
    params=scorebed_reported_params(policy.training_state,True)
    if args.journal_plan:
        from src.research.journal_scorebed_bridge import restored_params
        params=restored_params(params,args.journal_plan)
    check_tree(params,'policy parameters')
    from flax.traverse_util import flatten_dict
    arrays={'/'.join(path):np.asarray(value) for path,value in flatten_dict(params).items()}
    cpu=jax.devices('cpu')[0]
    with jax.default_device(cpu):
        cpu_params=jax.device_put(params,cpu);variables={'params':cpu_params}
        key=jax.random.key(args.seed+10000019);keys=jax.random.split(key,3)
        x=jnp.broadcast_to(jnp.linspace(5.,target.end-5,args.horizon)[None,:,None],(8,args.horizon,1))
        y=jax.random.uniform(keys[0],(8,args.horizon,1),dtype=jnp.float64)*target.population
        z=jax.random.normal(keys[1],(8,args.horizon,1),dtype=jnp.float64)
        carry=net.apply(variables,(8,),method='init_gru_state');expected=[]
        for t in range(args.horizon):
            raw=net.apply(variables,carry[-1],z=z[:,t],method='emitter_from_state')[:,0]
            previous=jnp.zeros((8,)) if t==0 else x[:,t-1,0]
            expected.append(np.asarray(target.feasible(raw,previous,t)[:,None]))
            token=net.apply(variables,y=y[:,t],x=x[:,t],method='encoder');carry,_=net.apply(variables,carry,token,method='gru_step')
        arrays.update(reference_x=np.asarray(x),reference_y=np.asarray(y),reference_noise=np.asarray(z),reference_actions=np.stack(expected,1))
        arrays['sir_configuration_json']=np.asarray(json.dumps(sir));arrays['policy_architecture']=np.asarray('official_ScoreBED_IqbalGRU_SIR_v1')
    np.savez(out/'scorebed_official_policy.npz',**arrays)
    from flax import serialization
    (out/'scorebed_official_score.msgpack').write_bytes(serialization.to_bytes(score._score_variables_from_state()))
    data=dict(training='official_ScoreBED_JAX_score_matching_and_score_EIG_gradient',upstream_commit=COMMIT,
        logical_primary_ODE_stage_evaluations=(initialization+score_spent+policy_spent)*args.horizon,minimum_primary_RK4_steps=(initialization+score_spent+policy_spent)*args.horizon*sir['integration_steps_per_stage'],solver_work_note='Lower bound on primal solver work; adaptive subdivisions and autodiff extra work are not claimed as exact FLOPs',
        sir_score_adapter_revision=2,potential_hidden_dim=cfg['score_network']['potential_hidden_dim'],score_dimension_compatibility='Pinned native potential_hidden_dim avoids scalar-data/scalar-time LayerNorm degeneracy; unchanged Transformer, loss, optimizers and training allocation',
        reported_policy_parameters='params_ema',random_streams='native derive_key score_trainer and policy_trainer/restart0',
        generated_training_trajectories=initialization+score_spent+policy_spent,training_forward_trajectories=initialization+score_spent+policy_spent,
        contrast_training_trajectories=0,training_trajectory_budget=budget,training_budget_unit='primary_episodes',
        initialization_trajectories=initialization,standardisation_trajectories=stats_count,
        score_training_trajectories=score_spent,policy_training_trajectories=policy_spent,
        budget_reconciliation=allocation if not tiny else dict(smoke=True),score_batch=batch,actual_score_updates=int(score.training_state.step),actual_policy_updates=int(policy.training_state.step),
        training_seconds=time.perf_counter()-start,history=records,checkpoint_selection='final',
        architecture='official_4layer_256dim_8head_RFF_conservative_transformer',policy_architecture='official_IqbalGRUNet_256_256_64_two_GRU_64_256_256',
        adaptations='1million total primary episodes including normalization and initialization; one fresh policy; published dynamical-system P1 99.75:0.25 allocation transferred to SIR; original batch1024; final partial minibatches; fast-sampler keyword-argument compatibility fix; no architecture or estimator substitutions',
        design_bounds=[0.,target.end],bounded_mapping='chronological feasible-time sigmoid with minimum spacing',sir_configuration=sir,
        runtime=dict(jax=jax.__version__,devices=[str(d) for d in jax.devices()]))
    if not args.journal_plan and data['generated_training_trajectories']!=budget:raise AssertionError('Episode accounting mismatch')
    from src.research.resource_ledger import with_training_ledger
    data=with_training_ledger(data,args.horizon)
    (out/'scorebed_official_training.json').write_text(json.dumps(data,indent=2))
    print('SCOREBED_OFFICIAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--journal-plan');ap.add_argument('--budget',type=int,default=1000000);ap.add_argument('--horizon',type=int,default=30);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--batch-size',type=int,default=1024);ap.add_argument('--smoke',action='store_true');ap.add_argument('--sir-config',required=True);run(ap.parse_args())
