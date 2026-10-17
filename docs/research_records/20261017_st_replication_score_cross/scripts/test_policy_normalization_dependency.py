"""No optimizer steps: test whether policy-target calibration is actually read."""
import os,sys,json,hashlib,time
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_st_replication_score_reverse_cross';S=C/'score_source'
sys.path[:0]=[str(S),str(S/'src/research/vendor/scorebed')]
import numpy as np,jax,jax.numpy as jnp
from src.research.score_frozen import restore,treehash
from src.research.scorebed_worker import CompatibleLocationFinding
from policy_learning.loaders import load_policy_trainer
from policy_learning.utils.func import noop
from policy_learning.eig_estimators import get_grad_estimate_fn
started=time.monotonic();path=C/'bundles/score102_policy103.json'
frozen,a,key,cfg,_=restore(path,CompatibleLocationFinding)
_,b,keyb,cfgb,_=restore(path,CompatibleLocationFinding)
# Poison the policy-target normalization only. The frozen score network's own
# normalization remains untouched and correct throughout this test.
for field in ['y_mean','y_std','xi_mean','xi_std']:
 setattr(b,field,jnp.full_like(getattr(b,field),jnp.nan))
pa,_=load_policy_trainer(key,cfg,a,grad_fn=noop,callbacks=[])
pb,_=load_policy_trainer(keyb,cfgb,b,grad_fn=noop,callbacks=[])
assert treehash(pa.training_state.params)==treehash(pb.training_state.params)
ds,xs=frozen.get_unbatched_score_fns()
grads=[];nextkeys=[]
for target,policy in [(a,pa),(b,pb)]:
 fn=get_grad_estimate_fn(policy.policy_network,target,ds,xs,N=1,outer_batch_size=1,M=1024,estimate_type='score')
 g,k=fn(jax.random.key(773103),policy.training_state.params)
 assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(g))
 grads.append(g);nextkeys.append(jax.random.key_data(k))
assert treehash(grads[0])==treehash(grads[1]),'Policy normalization affects native gradient; extra calibration remains relevant'
assert np.array_equal(nextkeys[0],nextkeys[1])
rec=dict(status='passed',native_score_gradient_identical_with_nan_policy_normalization=True,policy_parameters_identical=True,score_bundle_normalization_untouched=True,policy_target_normalization_not_a_training_dependency=True,new_initialization_simulations=20,new_gradient_diagnostic_episodes=2,optimizer_updates=0,device='cpu',cpu_affinity=sorted(os.sched_getaffinity(0)),seconds=time.monotonic()-started,gradient_sha256=treehash(grads[0]),code_audit='Native DADNet builder receives shape and feasible output activation, no std_stats; location reparam/sample/likelihood use physical units and do not read normalization fields. Only the frozen score network consumes its source calibration.')
(C/'policy_normalization_dependency_test.json').write_text(json.dumps(rec,indent=2)+'\n');print('POLICY_NORMALIZATION_NOT_A_TRAINING_DEPENDENCY',flush=True)
