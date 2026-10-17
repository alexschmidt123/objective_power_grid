import os,sys,json,hashlib
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_st_replication_score_reverse_cross';S=C/'score_source';Q=R/'experiments/10162026/10162026_location_root_cause_v4';OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget/location'
sys.path[:0]=[str(S),str(S/'src/research/vendor/scorebed')]
import jax,jax.numpy as jnp,numpy as np,yaml
from src.research.score_frozen import restore,target_from_stats,treehash
from src.research.scorebed_worker import CompatibleLocationFinding
from policy_learning.loaders import load_score_trainer,load_policy_trainer
from policy_learning.utils.func import noop
assert jax.devices()[0].platform=='gpu'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
cfgs={s:json.loads((OLD/f'seed{s}/scorebed/models/scorebed_official_config.json').read_text()) for s in [102,103]}
a=json.loads(json.dumps(cfgs[102]));b=json.loads(json.dumps(cfgs[103]));a['bed_model']['init_seed']=103;assert a==b,'Configurations differ beyond native initialization seed'
target=target_from_stats(CompatibleLocationFinding,103,Q/'bundles/normalization_gpu_103.npz')
# Native initialization follows published loader path; no score training.
unused,key=load_score_trainer(jax.random.key(103),cfgs[103],target,callbacks=[])
native,_=load_policy_trainer(key,cfgs[103],target,grad_fn=noop,callbacks=[])
initial=dict(parameters_sha256=treehash(native.training_state.params),trainer_key=jax.random.key_data(native.key).tolist(),state_rng=jax.random.key_data(native.training_state.rng).tolist())
bundle=json.loads((Q/'bundles/score_102.json').read_text())
bundle.update(policy_seed=103,policy_configuration=str(OLD/'seed103/scorebed/models/scorebed_official_config.json'),policy_normalization=str(Q/'bundles/normalization_gpu_103.npz'),policy_loader_key=jax.random.key_data(key).tolist(),initial_policy=initial)
paths=['score_configuration','policy_configuration','score_variables','score_normalization','policy_normalization']
bundle['sha256']={bundle[k]:sha(bundle[k]) for k in paths}
path=C/'bundles/score102_policy103.json';write(path,bundle)
score,restored_target,restored_key,restored_cfg,loaded=restore(path,CompatibleLocationFinding)
cross,_=load_policy_trainer(restored_key,restored_cfg,restored_target,grad_fn=noop,callbacks=[])
other=dict(parameters_sha256=treehash(cross.training_state.params),trainer_key=jax.random.key_data(cross.key).tolist(),state_rng=jax.random.key_data(cross.training_state.rng).tolist())
assert initial==other
assert treehash(target.get_std_stats_dict())==treehash(restored_target.get_std_stats_dict())
assert np.array_equal(jax.random.key_data(target.key),jax.random.key_data(restored_target.key))
assert sum([bundle['reused_pretraining_episodes'],bundle['new_policy_initialization_episodes'],bundle['new_policy_training_episodes']])==1000000
# Score model on feasible physical saved histories: loading both frozen paths
# must reproduce bundle102 predictions, with no normalization substitution.
original_score,*_=restore(Q/'bundles/score_102.json',CompatibleLocationFinding)
rows=json.loads((OLD/'seed103/scorebed/eval/scorebed_seed1001.json').read_text())[:2]
errors=[]
for r in rows:
 x=jnp.asarray(np.array(r['actions'],np.float32)[None]);y=jnp.asarray(np.array(r['observations'],np.float32)[None]);kw=dict(y=y,xi=x,sigma=jnp.ones(y.shape[:-2]),aux_data={},train=False)
 p=score.score_network.apply(score.variables,**kw);q=original_score.score_network.apply(original_score.variables,**kw)
 np.testing.assert_array_equal(np.asarray(p),np.asarray(q));assert np.isfinite(p).all();errors.append(float(np.max(abs(p-q))))
cfg=yaml.safe_load((Q/'original.yaml').read_text());cfg['research']['journal_protocol']['method_settings']['scorebed']['diagnostic_frozen_score_bundle']=str(path);cfg['research']['journal_protocol']['diagnostic_only']=True
(S/'configs/score102_policy103.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
write(C/'score_readiness.json',dict(status='passed',native_gpu=True,configuration_differences=['bed_model.init_seed'],initial_policy=initial,native_and_restored_policy_initialization_match=True,policy_normalization_and_target_rng_match=True,score_predictions_match_original_bundle102=True,prediction_max_errors=errors,preflight_primary_simulations=24,complete_model_episodes=1000000,new_primary_episodes=297006,normalization_recomputed=False,reserved_final_seed_used=False))
print('SCORE_REVERSE_CROSS_PREFLIGHT_PASSED',flush=True)
