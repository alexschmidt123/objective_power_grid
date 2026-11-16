"""Explicit reuse of a completed Fixed sequence to initialize a fresh MoE."""
import hashlib,json
from pathlib import Path
import torch

def load_fixed_initialization(source,args,metadata):
    source=Path(source).resolve()
    completion=json.loads((source/'completion.json').read_text())
    if completion.get('smoke_only') or 'fixed' not in completion['methods']:
        raise ValueError('Initialization requires a completed non-smoke Fixed run')
    if (source/'exit_code').read_text().strip()!='0':
        raise ValueError('Source run did not complete successfully')
    saved=json.loads((source/'run_config.json').read_text())
    for key in ('T','seed','N_obs','noise_sigma','window','observation_kind','duration_min',
                'duration_max','min_duration_separation','bus','amplitude',
                'contrasts','updates','batch_size','learning_rate','validation_systems','validate_every'):
        if saved['settings'].get(key)!=getattr(args,key):
            raise ValueError('Fixed initialization setting mismatch: '+key)
    if saved['physical_config']['swing_equation']!=json.loads(json.dumps(metadata['physical_config']['swing_equation'])):
        raise ValueError('Fixed initialization physics mismatch')
    for path,digest in saved['source_hashes'].items():
        if path.startswith('src/domains/swing/') or path=='src/objectives/eig/continuous_pathwise.py':
            if metadata['source_hashes'].get(path)!=digest:
                raise ValueError('Fixed initialization simulator/gradient source mismatch: '+path)
    path=source/'models/fixed.pth'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('method')!='fixed' or checkpoint.get('horizon')!=args.T:
        raise ValueError('Wrong Fixed checkpoint method or horizon')
    for key in ('seed','T','updates','batch_size','learning_rate','contrasts'):
        if checkpoint['settings'].get(key)!=saved['settings'].get(key):
            raise ValueError('Checkpoint settings mismatch: '+key)
    sequence=checkpoint['state_dict']['sequence'].detach().clone()
    if sequence.shape!=(args.T,) or not torch.isfinite(sequence).all():
        raise ValueError('Invalid Fixed sequence')
    return sequence,{'source_run':str(source),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_config_sha256':hashlib.sha256((source/'run_config.json').read_bytes()).hexdigest(),
        'purpose':'Starting sequence only; MoE network and optimizer train fresh.',
        'sequence_logits':sequence.tolist(),'settings_and_physics_verified':True,
        'same_simulator_and_pathwise_source':True}
