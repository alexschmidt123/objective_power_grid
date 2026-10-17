"""Frozen-score crossing preflight only: no optimizer steps or policy training."""
import os,sys,json,hashlib,time
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid')
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src/research/vendor/scorebed'))
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from src.research.scorebed_worker import CompatibleLocationFinding
from policy_learning.loaders import _build_score_network,load_score_trainer,load_policy_trainer
from policy_learning.utils.func import noop
C=R/'experiments/10162026/10162026_location_root_cause_v3'
OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget'
PREV=R/'experiments/10162026/10162026_location_stability_correction/scorebed_training_diagnosis'
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def treehash(tree):
    h=hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        x=np.asarray(x);h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes())
    return h.hexdigest()
def compare(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):return sum([compare(a[k],b[k],path+'/'+k) for k in sorted(set(a)&set(b))],[])+[path+'/'+k for k in set(a)^set(b)]
    return [] if a==b else [path]
def main():
    C.mkdir(exist_ok=True);bundles={};targets={};configs={};loaded={};shapes={};checks=[]
    rows=json.loads((OLD/'location/seed102/scorebed/eval/scorebed_seed1001.json').read_text())[:2]
    for seed in [102,103]:
        d=OLD/f'location/seed{seed}/scorebed/models';cfg=json.loads((d/'scorebed_official_config.json').read_text());configs[seed]=cfg
        target=CompatibleLocationFinding(T=30,d=2,k=2,init_seed=seed,sigma=.5,standardise=False,design_bounds=[-4,4])
        cached=C/f'score{seed}_normalization.npz'
        if cached.exists():
            frozen=np.load(cached)
            for key in ['y_mean','y_std','xi_mean','xi_std']:setattr(target,key,jnp.asarray(frozen[key]))
            target.standardised=True;target.key=jax.random.wrap_key_data(jnp.asarray(frozen['target_key']))
        else:target.compute_standardisation(10000)
        targets[seed]=target
        stats=target.get_std_stats_dict()
        # A restoration check avoids regenerating normalization in each future arm.
        restored=CompatibleLocationFinding(T=30,d=2,k=2,init_seed=seed,sigma=.5,standardise=False,design_bounds=[-4,4])
        for key,val in stats.items():setattr(restored,key,jnp.array(val))
        restored.standardised=True
        assert treehash(restored.get_std_stats_dict())==treehash(stats)
        net=_build_score_network(cfg,target);restore_net=_build_score_network(cfg,restored)
        variables=jax.tree_util.tree_map(jnp.asarray,serialization.msgpack_restore((d/'scorebed_official_score.msgpack').read_bytes()))
        leaves=jax.tree_util.tree_leaves(variables)
        assert all(np.isfinite(np.asarray(v)).all() for v in leaves)
        shapes[seed]=[(list(v.shape),str(v.dtype)) for v in leaves]
        saved=np.load(PREV/f'score_predictions_seed{seed}.npy')
        predict=jax.jit(lambda y,x:net.apply(variables,y=y,xi=x,
            sigma=jnp.ones(y.shape[:-2]),aux_data={},train=False))
        restore_predict=jax.jit(lambda y,x:restore_net.apply(variables,y=y,xi=x,
            sigma=jnp.ones(y.shape[:-2]),aux_data={},train=False))
        errors=[]
        for j,row in enumerate(rows):
            y=jnp.asarray(np.asarray(row['observations'],dtype=np.float32)[None]);x=jnp.asarray(np.asarray(row['actions'],dtype=np.float32)[None])
            value=predict(y,x);other=restore_predict(y,x)
            np.testing.assert_allclose(value,other,rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(np.asarray(value)[0],saved[j],rtol=1e-5,atol=1e-6)
            errors.append(float(np.max(abs(np.asarray(value)[0]-saved[j]))))
        np.savez(C/f'score{seed}_normalization.npz',**{k:np.asarray(v) for k,v in stats.items()},target_key=jax.random.key_data(target.key))
        bundles[str(seed)]=dict(config=str(d/'scorebed_official_config.json'),config_sha256=sha(d/'scorebed_official_config.json'),
            score_variables=str(d/'scorebed_official_score.msgpack'),score_sha256=sha(d/'scorebed_official_score.msgpack'),
            normalization_sha256=treehash(stats),normalization_file=str(C/f'score{seed}_normalization.npz'),
            previous_prediction_max_error=max(errors),normalization_restore_verified=True)
        loaded[seed]=variables
    assert shapes[102]==shapes[103]
    diffs=compare(configs[102],configs[103])
    assert set(diffs)<={'/bed_model/init_seed'},diffs
    # The native worker's root key returned by load_score_trainer is not
    # changed by the subsequent loop updating score.key. Reconstruct it once.
    score,policy_key=load_score_trainer(jax.random.key(102),configs[102],targets[102],callbacks=[])
    policies=[]
    for arm in [102,103]:
        p,_=load_policy_trainer(policy_key,configs[102],targets[102],grad_fn=noop,callbacks=[])
        policies.append(p)
        (C/f'cross_arm{arm}_initial_policy_parameters.msgpack').write_bytes(serialization.to_bytes(p.training_state.params))
    assert treehash(policies[0].training_state.params)==treehash(policies[1].training_state.params)
    assert np.array_equal(jax.random.key_data(policies[0].key),jax.random.key_data(policies[1].key))
    record=dict(status='compatibility_passed',training_launched=False,score_models=bundles,
        configuration_differences=diffs,variable_structure_matches=True,
        identical_policy_initial_parameters_sha256=treehash(policies[0].training_state.params),
        identical_policy_rng=jax.random.key_data(policies[0].key).tolist(),
        root_policy_loader_key=jax.random.key_data(policy_key).tolist(),
        original_initial_policy_hash_available=False,
        limitation='The pair is identical by construction and follows the native key path; no saved historical initial-policy hash exists for independent bitwise verification.',
        proposed_budget=dict(normalization_episodes=10000,score_initialization_episodes=4,score_training_episodes=692990,
            policy_initialization_episodes=10,policy_training_episodes=296996,total_episodes=1000000),
        new_preflight_simulation_episodes=4+20,
        prior_preflight_attempt_simulation_episodes=10000+20000+4+10,
        normalization_reused_from_preflight_cache=True,
        no_hyperparameter_or_architecture_changes=True,reserved_seed_used=False,
        training_gate='Preflight complete; report experiment design before launching any additional policy training beyond the existing three requested repeats.')
    write(C/'score_cross_preflight.json',record);print('SCORE_CROSS_PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback;write(C/'score_cross_preflight_failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
