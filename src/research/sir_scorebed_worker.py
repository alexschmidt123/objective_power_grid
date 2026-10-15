"""Pinned official ScoreBED training, with an explicit primary-episode cap."""
import os,sys,json,time,math,argparse,pathlib,subprocess
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('WANDB_MODE','disabled')
ROOT=pathlib.Path(__file__).resolve().parents[2]
VENDOR=ROOT/'src/research/vendor/scorebed'
sys.path.insert(0,str(VENDOR))
sys.path.insert(0,str(ROOT))
from src.research.sir_scorebed_model import SIRScoreModel,SIRScorePolicyNet
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
    initialization=stats_count+4+10
    budget=34 if tiny else args.budget
    remaining=budget-initialization
    if remaining<2:raise ValueError('ScoreBED cap cannot cover initialization')
    score_count=8 if tiny else int(.7*remaining)
    policy_count=remaining-score_count
    cfg=config(args.horizon,math.ceil(score_count/batch),batch)
    cfg['bed_model']['init_seed']=args.seed
    sir=json.loads(pathlib.Path(args.sir_config).read_text())
    cfg['bed_model']=dict(sir,name='sir_ode',T=args.horizon)
    (out/'scorebed_environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'scorebed_official_config.json').write_text(json.dumps(cfg,indent=2))
    # Use the official standardisation routine and charge its simulations.
    target=SIRScoreModel(T=args.horizon,init_seed=args.seed,**sir)
    target.compute_standardisation(stats_count)
    key=jax.random.key(args.seed)
    score,key=load_score_trainer(key,cfg,target,callbacks=[])
    score_spent=0;records=[]
    while score_spent<score_count:
        n=min(batch,score_count-score_spent)
        y,x,theta,aux,score.key=score._sample_target(score.key,(n,),use_fast_sampler=True)
        score.training_state,metrics=score.train_step(score.training_state,(y,x,theta,aux))
        # Synchronize for finite checks and honest wall timing.
        check_tree(metrics,'score metrics')
        score_spent+=n;step=int(score.training_state.step)
        if step%50==0 or score_spent==score_count:
            row=dict(phase='score',step=step,episodes=score_spent,metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1})
            records.append(row);print('ScoreBED official',row,flush=True)
    check_tree(score.training_state.params,'score parameters')
    data_score,design_score=score.get_unbatched_score_fns()
    net=SIRScorePolicyNet(encoder_shapes=[256],encoding_dim=16,enc_act='relu',emit_act='relu',emitter_shapes=[],shape=(args.horizon,1),zero_init_final_layer=True,time_scale=target.end,count_scale=target.population)
    policy,key=load_policy_trainer(key,cfg,target,grad_fn=noop,callbacks=[],policy_net=net)
    policy_spent=0;last_n=None
    while policy_spent<policy_count:
        n=min(batch,policy_count-policy_spent)
        if n!=last_n:
            grad_fn=get_grad_estimate_fn(policy.policy_network,target,data_score,design_score,N=n,outer_batch_size=n,M=1024,estimate_type='score')
            policy.update_grad_fn(grad_fn);last_n=n
        policy.training_state,metrics,_=policy.train_step(policy.training_state)
        check_tree(metrics,'policy metrics')
        policy_spent+=n;step=int(policy.training_state.step)
        if step%50==0 or policy_spent==policy_count:
            row=dict(phase='policy',step=step,episodes=policy_spent,metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1})
            records.append(row);print('ScoreBED official',row,flush=True)
    params=policy.training_state.params
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
        x=jnp.broadcast_to(jnp.linspace(5.,target.end-5,args.horizon)[None,:,None],(8,args.horizon,1));key,sub=jax.random.split(key)
        y=jax.random.uniform(sub,(8,args.horizon,1),dtype=jnp.float64)*target.population;enc=jnp.ones((8,16),dtype=jnp.float32)*.01
        expected=[]
        for t in range(args.horizon):
            raw=policy.policy_network.apply({'params':cpu_params},enc=enc,method='emitter')[:,0]
            previous=jnp.zeros((8,)) if t==0 else x[:,t-1,0]
            expected.append(np.asarray(target.feasible(raw,previous,t)[:,None]))
            enc=enc+policy.policy_network.apply({'params':cpu_params},y=y[:,t],x=x[:,t],method='encoder')
        arrays.update(reference_x=np.asarray(x),reference_y=np.asarray(y),reference_actions=np.stack(expected,1))
        arrays['sir_configuration_json']=np.asarray(json.dumps(sir))
    np.savez(out/'scorebed_official_policy.npz',**arrays)
    from flax import serialization
    (out/'scorebed_official_score.msgpack').write_bytes(serialization.to_bytes(score._score_variables_from_state()))
    data=dict(training='official_ScoreBED_JAX_score_matching_and_score_EIG_gradient',upstream_commit=COMMIT,
        generated_training_trajectories=initialization+score_spent+policy_spent,training_forward_trajectories=initialization+score_spent+policy_spent,
        contrast_training_trajectories=0,training_trajectory_budget=budget,training_budget_unit='primary_episodes',
        initialization_trajectories=initialization,standardisation_trajectories=stats_count,
        score_training_trajectories=score_spent,policy_training_trajectories=policy_spent,
        score_batch=batch,actual_score_updates=int(score.training_state.step),actual_policy_updates=int(policy.training_state.step),
        training_seconds=time.perf_counter()-start,history=records,checkpoint_selection='final',
        architecture='official_4layer_256dim_8head_RFF_conservative_transformer',policy_architecture='official_DADNet_256_16_sum_encoder',
        adaptations='1million total primary episodes including normalization and initialization; one fresh policy; 70percent remaining budget score pretraining; smaller minibatches under capped budget; final partial minibatches; fast-sampler keyword-argument compatibility fix; no architecture or estimator substitutions',
        design_bounds=[0.,target.end],bounded_mapping='chronological feasible-time sigmoid with minimum spacing',sir_configuration=sir,
        runtime=dict(jax=jax.__version__,devices=[str(d) for d in jax.devices()]))
    if data['generated_training_trajectories']!=budget:raise AssertionError('Episode accounting mismatch')
    (out/'scorebed_official_training.json').write_text(json.dumps(data,indent=2))
    print('SCOREBED_OFFICIAL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--budget',type=int,default=1000000);ap.add_argument('--horizon',type=int,default=30);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--smoke',action='store_true');ap.add_argument('--sir-config',required=True);run(ap.parse_args())
