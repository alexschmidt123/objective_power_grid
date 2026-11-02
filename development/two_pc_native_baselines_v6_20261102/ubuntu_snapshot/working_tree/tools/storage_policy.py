"""Storage boundary checks; this module never starts experiments."""
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

def _option(argv, name):
    for i, arg in enumerate(argv):
        if arg == name:
            if i + 1 == len(argv):
                raise ValueError(name + ' requires a value')
            return argv[i + 1]
        if arg.startswith(name + '='):
            return arg.split('=', 1)[1]
    return None

def enforce_storage_policy(root, argv):
    root = Path(root).resolve()
    argv = list(argv)
    if '--help' in argv or '-h' in argv:
        return argv
    for option in ('--output', '--campaign-root'):
        count = sum(arg.split('=', 1)[0] == option for arg in argv)
        if count > 1:
            raise ValueError('Duplicate storage option: ' + option)
    for arg in argv:
        flag = arg.split('=', 1)[0]
        if flag.startswith('--out') and flag != '--output':
            raise ValueError('Use the full --output option')
        if flag.startswith('--campaign') and flag != '--campaign-root':
            raise ValueError('Use the full --campaign-root option')
    if root.name not in ('objective_power_grid', 'objective_power_grid_temp'):
        raise ValueError('Experiments must execute from the STABLE or TEMP project root')
    sweep = '--sweep' in argv
    flag = '--campaign-root' if sweep else '--output'
    value = _option(argv, flag)
    if root.name == 'objective_power_grid_temp':
        scratch = (root / 'scratch_runs').resolve()
        if value is None:
            date = datetime.now().strftime('%m%d%Y')
            value = str(scratch / date / (date + '_' + uuid.uuid4().hex))
            argv += [flag, value]
        path = Path(value)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if path == scratch or scratch not in path.parents:
            raise ValueError('TEMP experiment output must be beneath TEMP/scratch_runs/')
        # Reject an unused explicit output option on a sweep too.
        extra = _option(argv, '--output') if sweep else None
        if extra is not None:
            raise ValueError('Sweeps use --campaign-root, not --output')
        if path.exists():
            raise ValueError('Output already exists; use a new run identifier')
        for i, arg in enumerate(argv):
            if arg == flag:
                argv[i + 1] = str(path)
            elif arg.startswith(flag + '='):
                argv[i] = flag + '=' + str(path)
        return argv
    if sweep:
        raise ValueError('Official sweeps require separately frozen per-run authorizations')
    if value is None:
        raise ValueError('STABLE runs require an explicit benchmark output and frozen authorization')
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not any((root / 'experiments' / b).resolve() in path.parents for b in ('location_finding', 'sir_ode')):
        raise ValueError('Official output must be under experiments/location_finding/ or experiments/sir_ode/')
    if path.exists() or '--force' in argv:
        raise ValueError('Official run identifiers cannot be overwritten')
    key = hashlib.sha256(json.dumps(argv, separators=(',', ':')).encode()).hexdigest()
    auth = root / 'reports/official_run_authorizations' / (key + '.json')
    if not auth.is_file():
        raise ValueError('Missing frozen official-run authorization: ' + str(auth))
    record = json.loads(auth.read_text())
    if record.get('status') != 'VALIDATED_FROZEN_SOURCE' or record.get('regression_status') != 'PASS' or record.get('argv') != argv:
        raise ValueError('Official authorization lacks validated source, passing regressions, or exact command')
    manifest = record.get('source_sha256', {})
    required = {str(p.relative_to(root)) for prefix in ('src', 'tools', 'configs', 'scripts') for p in (root / prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.yaml', '.yml', '.sh')}
    required.update(('run.sh', 'sweep_run.sh'))
    if not required.issubset(manifest):
        raise ValueError('Official authorization does not freeze every source/configuration file')
    for rel, digest in manifest.items():
        p = (root / rel).resolve()
        if root not in p.parents or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source mismatch: ' + rel)
    return argv
