"""CER isolation and read-only historical dependencies."""
import os, sys, json, time, math, pathlib, hashlib, copy
sys.dont_write_bytecode = True
import numpy as np
import torch
ROOT = pathlib.Path.home() / 'Documents/objective_power_grid_temp'
R = ROOT / 'scratch_runs'
OLD = R / 'moe_hec_v2_icml_20261109'
O = pathlib.Path(os.environ.get('CER_ROOT', str(R / 'cer_moe_sboed_v3_20261111')))
os.environ['MOE_V2_RUN_ROOT'] = str(OLD)
sys.path.insert(0, str(OLD / 'source'))
import restore_v1, train_v2
from native_common import original, sha, digest, allopts, norm, P, cloudpickle
from normalized_reward import normalize_native_post
from hec_policy import mean_signal, advance, COST
from soft_policy import prefix_inputs
CFG = json.loads((O / 'CONFIG_LOCKED.json').read_text())

def put(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def cv(v):
        if isinstance(v, torch.Tensor): return v.detach().cpu().tolist()
        if hasattr(v, 'item'): return v.item()
        raise TypeError(type(v).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=cv) + '\n')
    tmp.replace(path)

def restore(device='cuda'):
    import io
    receipt = json.loads((OLD / 'arms/candidate/RESULT.json').read_text())
    runtime = OLD / 'arms/candidate/runtime.pkl'
    assert sha(runtime) == receipt['runtime_sha256']
    assert sha(OLD / 'arms/candidate/actor.pt') == receipt['actor_sha256']
    loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
    try:
        with runtime.open('rb') as f: state = cloudpickle.load(f)
    finally: torch.storage._load_from_bytes = loader
    a = state['algorithm']
    assert a.update == 16001 and a.discount == 1 and not a.use_PCE
    assert a.n_batch == 10000 and len(a._runtime_buffer['params']) == 1000000
    for opt in allopts(a):
        opt._optimizer_step_pre_hooks.clear(); opt._optimizer_step_post_hooks.clear()
    normalize_native_post(a)
    return state

def protected():
    paths = []
    for name in ['moe_hec_v2_icml_20261109', 'moe_specialization_guided_eig_20261108',
                 'moe_sboed_v3', 'moe_sboed_v3_competitive_20261110',
                 'moe_sboed_v3_model_informed_20261110', 'moe_hec_v21_joint_learning']:
        base = R / name
        paths.extend(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and (p.suffix in ['.py', '.pt', '.pkl'] or p.name in ['CONFIG_LOCKED.json', 'PRETRAIN_MANIFEST.json']))
    return {str(p): sha(p) for p in paths}
