#!/usr/bin/env python3
"""Submit a frozen IEEE9 master-bank audit with explicit reusable inputs.

No submission or filesystem mutation occurs on import or --help.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile


def extract_source(archive, destination):
    """Accept ordinary repository files only, excluding traversal and links."""
    with tarfile.open(str(archive)) as source:
        for member in source.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsupported archive member: ' + member.name)
        source.extractall(str(destination))
    for relative in ('run.sh', 'scripts/audit.sh', 'scripts/check.sh',
                     'hprc/master_audit.slurm', 'configs/ieee9_mocu.yaml'):
        if not (destination / relative).is_file():
            raise ValueError('Incomplete source archive: ' + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--runtime', type=Path,
                        default=Path('/scratch/user/g.lin/objective_power_grid_runtime'))
    parser.add_argument('--previous', type=Path, help='Optional previous campaign JSON for candidate reuse')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--partition', default='gpu')
    parser.add_argument('--gres', default='gpu:a100:1')
    args = parser.parse_args()
    if args.concurrency < 1 or not args.archive.is_file():
        parser.error('Require a source archive and positive concurrency')
    if args.previous is not None:
        previous = json.loads(args.previous.read_text())
        if not isinstance(previous.get('candidates'), list):
            parser.error('Previous campaign must contain a candidates list')
    runtime = args.runtime.resolve()
    if not (runtime / 'data').is_dir():
        parser.error('Runtime data directory is missing')
    stamp = datetime.datetime.now().strftime('%m%d%Y_%H%M%S_%f')
    out = runtime / 'experiments' / (stamp + '_ieee9_master_search_Uctrl_T5_Nobs5_sigma0p005')
    source = out / 'source_snapshot'
    source.mkdir(parents=True, exist_ok=False)
    extract_source(args.archive, source)
    (source / 'data').symlink_to(runtime / 'data', target_is_directory=True)
    if args.previous is not None:
        shutil.copy2(str(args.previous), str(out / 'previous_campaign.json'))
    files = {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
             for base in ('src', 'tools', 'configs', 'scripts', 'hprc')
             for f in (source / base).rglob('*') if f.is_file()}
    for name in ('run.sh', 'sweep_run.sh', 'AGENTS.md'):
        if (source / name).is_file():
            files[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    (out / 'source_manifest.json').write_text(json.dumps({
        'commit': args.commit, 'files': files,
        'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        'previous': str(args.previous) if args.previous else None,
    }, indent=2) + '\n')
    logs = out / 'logs'
    logs.mkdir()
    jobs = {}

    def submit(stage, options):
        command = ['sbatch', '--parsable', '--job-name=ieee9-master-' + stage,
                   '--output=' + str(logs / '%x-%A-%a.out')]
        command += options + [str(source / 'hprc/master_audit.slurm'), stage, str(out)]
        job = subprocess.check_output(command, universal_newlines=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected job ID: ' + job)
        jobs[stage] = {'job_id': job, 'command': command}
        (out / 'slurm_jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
        print(stage, job, flush=True)
        return job

    gpu = ['--partition=' + args.partition, '--gres=' + args.gres]
    prep = submit('prepare', gpu + ['--time=12:00:00'])
    fresh = submit('fresh', gpu + ['--time=02:00:00', '--dependency=afterok:' + prep])
    confirm = submit('confirm', gpu + ['--time=12:00:00',
        '--array=0-4%' + str(args.concurrency), '--dependency=afterok:' + fresh])
    submit('finalize', ['--partition=short', '--time=00:15:00', '--dependency=afterok:' + confirm])
    print('CAMPAIGN=' + str(out), flush=True)


if __name__ == '__main__':
    main()
