"""Immutable original V2 evaluation identity; no generated replacement banks."""
import hashlib,importlib.util,pathlib,sys
sys.dont_write_bytecode=True
import numpy as np
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
BASE=ROOT/'scratch_runs/moe_hec_v2_icml_20261109'
BANK=BASE/'evaluation/development_validation_bank.npz'
UNIFORMS=BASE/'evaluation/routing_uniforms.npy'
SCORER=BASE/'source/frozen_evaluator.py'
LOCK=dict(cases=2000,contrasts=1000000,horizon=30,sigma=.5,chunk=16384,
    bank_sha256='52c0c614c4ee3b63d58ccda60714b38b00a2c0ca35564a428fe8cc2ce8d727c5',
    uniforms_sha256='6e56f0a08fe4deeefd7f0977967dba9d2910a4a383f1acfbd1736db3f358033c',
    scorer_sha256='64c67b6547c77f4823b487493ed0703efbd3f04d7308019e875d145505695f2f')
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8388608),b''):h.update(part)
    return h.hexdigest()
def validate(identity):
    if identity!=LOCK:raise ValueError('Primary comparison must use exact original V2 protocol; altered bank/count/scorer rejected')
def load():
    validate(LOCK)
    assert sha(BANK)==LOCK['bank_sha256'] and sha(UNIFORMS)==LOCK['uniforms_sha256'] and sha(SCORER)==LOCK['scorer_sha256']
    bank=np.load(BANK);u=np.load(UNIFORMS)
    assert bank['truth'].shape==(2000,4) and bank['noise'].shape==(2000,30) and bank['contrasts'].shape==(1000000,4) and u.shape==(2000,30)
    spec=importlib.util.spec_from_file_location('canonical_v2_scorer',SCORER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return bank,u,mod.contrast_scores
def tests():
    load();rejected=[]
    for key,value in [('cases',1024),('contrasts',16000000),('bank_sha256','changed'),('scorer_sha256','changed')]:
        altered=dict(LOCK);altered[key]=value
        try:validate(altered)
        except ValueError:rejected.append(key)
    assert len(rejected)==4
    return dict(status='PASS',original_bank_shapes_and_hashes=True,rejected_primary_changes=rejected)
