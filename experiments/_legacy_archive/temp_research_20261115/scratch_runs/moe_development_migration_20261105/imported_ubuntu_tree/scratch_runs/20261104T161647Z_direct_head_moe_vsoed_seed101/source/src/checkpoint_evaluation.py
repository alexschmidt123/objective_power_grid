"""Explicit evaluation-only reuse of completed, scientifically matching policies."""
import ast
import hashlib
import json
from pathlib import Path

EVALUATION_SETTINGS = {'methods', 'eval_systems'}

RUNTIME_SETTINGS = {'config', 'output', 'eval_seed', 'eval_seeds',
                    'preflight_max_hours', 'estimate_only', 'evaluate_from'}

def read_source(directory):
    source = Path(directory).expanduser().resolve()
    config = json.loads((source / 'run_config.json').read_text())
    done = json.loads((source / 'completion.json').read_text())
    if config.get('evaluation_only'):
        raise ValueError('Use the original training run, not an evaluation-only output')
    if (done.get('objective') != config['settings'].get('objective') or
            set(done.get('methods', [])) != set(config['settings']['methods'].split(',')) or
            done.get('records', 0) <= 0 or not (source / 'summary.json').is_file()):
        raise ValueError('Source must be a completed compatible training/evaluation run')
    if (source / 'failure.json').exists() or (source / 'preflight_blocked.json').exists():
        raise ValueError('Cannot reuse a failed or preflight-blocked run')
    if (source / 'exit_code').exists() and (source / 'exit_code').read_text().strip() != '0':
        raise ValueError('Source run did not exit successfully')
    snapshot = source / 'source_snapshot' / 'configs' / Path(config['settings']['config']).name
    if not snapshot.is_file():
        raise ValueError('Source configuration snapshot is missing')
    return source, config, done, snapshot

def inherit_settings(parser):
    known, _ = parser.parse_known_args()
    if not known.evaluate_from:
        return None
    source, config, done, snapshot = read_source(known.evaluate_from)
    allowed = {a.dest for a in parser._actions}
    defaults = {k: v for k, v in config['settings'].items()
                if k in allowed and k not in RUNTIME_SETTINGS}
    parser.set_defaults(**defaults, config=str(snapshot))
    import sys
    if not any(a.split('=')[0] in ('--eval-seeds', '--eval-seed') for a in sys.argv[1:]):
        parser.error('--evaluate-from requires explicit additional --eval-seeds or --eval-seed')
    return source, config, done

def scientific_ast(path):
    tree = ast.parse(Path(path).read_text())
    # main handles CLI orchestration; scientific definitions remain exact.
    tree.body = [node for node in tree.body
                 if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'main')
                 and not (isinstance(node, ast.If) and ast.unparse(node.test) in ("__name__ == '__main__'", "'__main__' == __name__"))]
    return ast.dump(tree, include_attributes=False)

def verify_source_code(root, source, config):
    hashes = config.get('source_hashes', {})
    if not hashes or 'src/objectives/eig/continuous_eig.py' not in hashes:
        raise ValueError('Source lacks scientific implementation provenance')
    for relative, expected in hashes.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'src':
            raise ValueError('Invalid source provenance path: ' + relative)
        current = Path(root) / rel
        if not current.is_file():
            raise ValueError('Missing implementation: ' + relative)
        if hashlib.sha256(current.read_bytes()).hexdigest() == expected:
            continue
        if relative == 'src/online_cli.py':
            continue  # dispatch only; scientific modules are checked separately
        if relative == 'src/objectives/eig/continuous_eig.py':
            old = source / 'source_snapshot' / rel
            if (old.is_file() and hashlib.sha256(old.read_bytes()).hexdigest() == expected
                    and scientific_ast(old) == scientific_ast(current)):
                continue
        raise ValueError('Scientific implementation mismatch: ' + relative)

def validate_reuse(reuse, args, root, eval_seeds):
    source, config, done = reuse
    differences = [k for k, v in config['settings'].items()
                   if k not in RUNTIME_SETTINGS | EVALUATION_SETTINGS and getattr(args, k, None) != v]
    if differences:
        raise ValueError('Checkpoint settings mismatch: ' + ', '.join(differences))
    if args.estimate_only or args.preflight_max_hours:
        raise ValueError('Evaluation-only reuse does not support training preflight/estimate flags')
    requested = set(args.methods.split(','))
    available = set(config['settings']['methods'].split(','))
    if available & {'dad', 'step_dad'}:
        available |= {'dad', 'step_dad'}
    if not requested or not requested <= available:
        raise ValueError('Requested methods lack matching source policies')
    if getattr(args, 'eval_systems', 1) < 1:
        raise ValueError('eval_systems must be positive')
    if (set(eval_seeds) & set(done['evaluation_seeds']) and
            requested & set(done['methods']) and
            getattr(args, 'eval_systems', None) == config['settings'].get('eval_systems')):
        raise ValueError('Request additional evaluation seeds or a different test-set size for previously evaluated methods')
    verify_source_code(root, source, config)

def load_policies(reuse, args, engine, metadata, policy_class, redq_class):
    import torch
    source, config, _ = reuse
    for field in ('protocol', 'physical_config', 'observation_kind', 'duration_order'):
        actual = json.loads(json.dumps(metadata.get(field)))
        expected = json.loads(json.dumps(config.get(field)))
        # Method selection is launch metadata embedded in the resolved config;
        # all physical, observation, horizon and constraint fields remain exact.
        if field == 'physical_config':
            for value in (actual, expected):
                if isinstance(value, dict) and isinstance(value.get('experiment'), dict):
                    value['experiment'].pop('methods', None)
        if actual != expected:
            raise ValueError('Checkpoint experiment mismatch: ' + field)
    methods = set(args.methods.split(','))
    if 'step_dad' in methods:
        methods.add('dad')
    policies, checkpoints = {}, {}
    for method in ('fixed', 'dad', 'rl_sboed'):
        if method not in methods:
            continue
        path = source / 'models' / (method + '.pth')
        blob = torch.load(path, map_location='cpu', weights_only=True)
        if (blob.get('method') != method or blob.get('horizon') != args.T or
                blob.get('n_obs') != engine.n_obs or blob.get('protocol') != metadata['protocol']):
            raise ValueError('Checkpoint identity mismatch: ' + method)
        for key, value in config['settings'].items():
            if key not in RUNTIME_SETTINGS and blob.get('settings', {}).get(key) != value:
                raise ValueError('Checkpoint saved settings mismatch: ' + key)
        diagnostics = json.loads((source / 'models' / (method + '_training_diagnostics.json')).read_text())
        if diagnostics.get('completed_updates') != args.updates:
            raise ValueError('Checkpoint training is incomplete: ' + method)
        policy = redq_class(args.T, engine.n_obs) if method == 'rl_sboed' else policy_class(args.T, engine.n_obs, fixed=method == 'fixed')
        policy.load_state_dict(blob['state_dict'], strict=True)
        policy.eval()
        if any(not torch.isfinite(p).all() for p in policy.parameters()):
            raise ValueError('Checkpoint contains nonfinite parameters: ' + method)
        policies[method] = policy
        checkpoints[method] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    provenance = {'training_run': str(source), 'training_seed': args.seed,
                  'source_config_sha256': hashlib.sha256((source / 'run_config.json').read_bytes()).hexdigest(),
                  'checkpoints': checkpoints,
                  'evaluation_overrides': {k: {'source': config['settings'].get(k), 'evaluation': getattr(args, k, None)}
                      for k in EVALUATION_SETTINGS if config['settings'].get(k) != getattr(args, k, None)},
                  'source_code_check': 'exact hashes; continuous_eig scientific AST permits CLI-only changes; online_cli dispatch excluded'}
    return policies, provenance
