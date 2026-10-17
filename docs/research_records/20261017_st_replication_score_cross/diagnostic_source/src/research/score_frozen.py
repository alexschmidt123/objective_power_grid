"""Restore a complete frozen native score model without retraining it."""
import json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from policy_learning.loaders import _build_score_network
from policy_learning.trainers.train_scores import ScoreTrainer


def treehash(tree):
    h=hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        v=np.asarray(x);h.update(str((v.shape,str(v.dtype))).encode());h.update(v.tobytes())
    return h.hexdigest()


def target_from_stats(factory,seed,path):
    target=factory(T=30,d=2,k=2,init_seed=seed,sigma=.5,standardise=False,design_bounds=[-4,4])
    z=np.load(path)
    for key in ['y_mean','y_std','xi_mean','xi_std']:setattr(target,key,jnp.asarray(z[key]))
    target.standardised=True
    target.key=jax.random.wrap_key_data(jnp.asarray(z['target_key']))
    return target


class FrozenScore(ScoreTrainer):
    def __init__(self,cfg,target,variables,updates):
        self.score_network=_build_score_network(cfg,target)
        self.target=target;self.variables=variables
        self.training_state=SimpleNamespace(params=variables['params'],step=updates)
    def _score_variables_from_state(self):return self.variables


def restore(bundle_path,factory):
    bundle=json.loads(Path(bundle_path).read_text())
    for path,digest in bundle['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    source_cfg=json.loads(Path(bundle['score_configuration']).read_text())
    policy_cfg=json.loads(Path(bundle['policy_configuration']).read_text())
    source_target=target_from_stats(factory,bundle['score_seed'],bundle['score_normalization'])
    policy_target=target_from_stats(factory,bundle['policy_seed'],bundle['policy_normalization'])
    variables=jax.tree_util.tree_map(jnp.asarray,serialization.msgpack_restore(Path(bundle['score_variables']).read_bytes()))
    frozen=FrozenScore(source_cfg,source_target,variables,bundle['score_updates'])
    key=jax.random.wrap_key_data(jnp.asarray(bundle['policy_loader_key'],dtype=jnp.uint32))
    return frozen,policy_target,key,policy_cfg,bundle
