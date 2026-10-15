"""Explicit, verified evaluation reuse of a completed upstream vsOED actor."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from src.research.upstream_vsoed import UpstreamActor,COMMIT,compress_input
from src.research.vendor.vsoed.utils import Net


def load_actor(source,root,raw,horizon,seed):
    source=Path(source).resolve();root=Path(root)
    if (source/'exit_code').read_text().strip()!='0':raise ValueError('Source run must be complete')
    config=json.loads((source/'run_config.json').read_text())
    if config['horizon']!=horizon or config['train_seed']!=seed or config['smoke']:
        raise ValueError('Source horizon/training seed/protocol mismatch')
    old=config['benchmark']
    for key in ['system','observation','location_finding']:
        if old.get(key)!=raw.get(key):raise ValueError(f'Source benchmark mismatch: {key}')
    if old['research']['upstream_vsoed']!=raw['research']['upstream_vsoed']:
        raise ValueError('Upstream training profile mismatch')
    manifest=json.loads((source/'source_manifest.json').read_text())['sha256']
    for name,expected in manifest.items():
        if name.startswith('src/research/vendor/vsoed/') and (root/name).is_file():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Upstream source changed: {name}')
    def definition(path,name):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name),include_attributes=False)
    for path,names in [('src/research/benchmarks.py',['Benchmark','LocationFinding']),
                       ('src/research/policies.py',['history_features']),
                       ('src/research/upstream_vsoed.py',['UpstreamActor'])]:
        for name in names:
            if definition(root/path,name)!=definition(source/'source'/path,name):
                raise ValueError(f'Scientific evaluation interface changed: {name}')
    checkpoint=source/'models/vsoed.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training=json.loads((source/'models/vsoed_training.json').read_text())
    if training['checkpoint_sha256']!=digest:raise ValueError('Checkpoint checksum mismatch')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['commit']!=COMMIT or saved['horizon']!=horizon or saved['implementation']!='upstream_GMM_I':
        raise ValueError('Unsupported checkpoint')
    bounds=[[-float('inf'),float('inf')]]*2 if raw.get('location_finding',{}).get('design_bounds',[-4,4]) is None else [raw['location_finding']['design_bounds']]*2
    actor=Net([horizon+3*(horizon-1),256,256,256,2],nn.ReLU,
              torch.tensor(bounds,dtype=torch.float32),'actor').float()
    actor.load_state_dict(saved['actor_state']);actor.eval()
    if saved.get('network_input_compression')=='asinh':
        actor.register_forward_pre_hook(compress_input)
    record=dict(training='evaluation_only',training_seconds=0.,policy_parameters=sum(p.numel() for p in actor.parameters()),
                source_run=str(source),source_checkpoint_sha256=digest,source_training=training,
                fresh_training=False,upstream_commit=COMMIT)
    return UpstreamActor(actor,horizon),record
