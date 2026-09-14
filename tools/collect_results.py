"""Reusable lightweight result manifests and verified copies; never load models."""
import argparse
import json
from pathlib import Path
import re
import shutil
if __package__:
    from .archive_records import digest, safe_path, verify
else:
    from archive_records import digest, safe_path, verify

EXCLUDED_DIRS={'source','source_snapshot','banks','data','.venv','__pycache__','.cache'}
EXCLUDED_SUFFIXES={'.pt','.pth','.ckpt','.bin','.npy','.npz','.h5','.hdf5','.pkl','.pickle','.tar','.gz','.zip','.safetensors'}


def lightweight_manifest(project, campaign, max_bytes=50*1024*1024):
    """Keep project-relative names; retain JSON histories inside models/."""
    project=Path(project).resolve();relative=Path(campaign)
    if len(relative.parts)<3 or relative.parts[0]!='experiments' or not re.fullmatch(r'\d{8}',relative.parts[1]):
        raise ValueError('Use experiments/MMDDYYYY/campaign as the relative campaign path')
    base=safe_path(project,relative.as_posix())
    if not base.is_dir() or base.is_symlink():raise ValueError('Campaign must be a real directory')
    entries=[];excluded=[]
    for p in sorted(base.rglob('*')):
        rel=p.relative_to(project).as_posix()
        if p.is_symlink():
            excluded.append({'path':rel,'reason':'symlink'});continue
        if not p.is_file():continue
        reason=None
        if set(p.relative_to(base).parts[:-1]) & EXCLUDED_DIRS:reason='source/data/cache'
        elif p.suffix.lower() in EXCLUDED_SUFFIXES:reason='large binary/archive'
        elif p.stat().st_size>max_bytes:reason='size limit; review if essential result table'
        if reason:
            excluded.append({'path':rel,'reason':reason});continue
        with p.open('rb') as stream:sha=digest(stream)
        entries.append({'path':rel,'kind':'file','sha256':sha,'bytes':p.stat().st_size})
    return {'schema':'lightweight_results_v1','campaign':relative.as_posix(),
            'entries':entries,'excluded':excluded}


def copy_results(source_project,destination_project,manifest):
    """Copy selected artifacts without renaming or overwriting different content."""
    source=Path(source_project).resolve();destination=Path(destination_project).resolve()
    if source==destination:raise ValueError('Source and destination must differ')
    verify(source,manifest)
    # Reject all collisions before copying any files.
    for entry in manifest['entries']:
        target=safe_path(destination,entry['path'])
        if target.is_symlink():raise ValueError('Destination symlink: '+str(target))
        if target.exists():
            if not target.is_file():raise ValueError('Destination is not a file: '+str(target))
            with target.open('rb') as stream:
                if digest(stream)!=entry['sha256']:raise ValueError('Different destination content: '+str(target))
    for entry in manifest['entries']:
        target=safe_path(destination,entry['path'])
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(safe_path(source,entry['path']),target)
    verify(destination,manifest)
    return {'status':'verified','source':str(source),'destination':str(destination),
            'campaign':manifest['campaign'],'verified_files':len(manifest['entries']),
            'excluded':manifest['excluded']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-project',required=True)
    p.add_argument('--campaign',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--destination-project')
    p.add_argument('--receipt')
    args=p.parse_args()
    if args.destination_project and not args.receipt:p.error('Copy requires --receipt')
    doc=lightweight_manifest(args.source_project,args.campaign)
    Path(args.manifest).write_text(json.dumps(doc,indent=2)+'\n')
    if args.destination_project:
        receipt=copy_results(args.source_project,args.destination_project,doc)
        Path(args.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    print('RESULT_MANIFEST_COMPLETE',len(doc['entries']),'files;',len(doc['excluded']),'excluded')

if __name__=='__main__':main()
