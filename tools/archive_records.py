"""Verify an experiment archive before retiring its original files.

No bank directories are deleted. Symlinks are checked without following them.
Use manifest on a tar archive, verify on the extracted copy, then prune on the
source only after saving the successful destination verification receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'experiments':
        raise ValueError('Unsafe archive path: ' + name)
    path = root.joinpath(*p.parts)
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Symlink parent: ' + str(parent))
    return path


def manifest(archive):
    entries = []
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.isdir():
                continue
            safe_path(Path('/unused-archive-root'), member.name)
            if member.issym():
                entry = {'path': member.name, 'kind': 'symlink', 'target': member.linkname}
            elif member.isfile() or member.islnk():
                with tar.extractfile(member) as stream:
                    entry = {'path': member.name, 'kind': 'file', 'sha256': digest(stream)}
            else:
                raise ValueError('Unsupported archive entry: ' + member.name)
            entries.append(entry)
    return {'schema': 'experiment_archive_v1', 'entries': entries}


def verify(root, doc):
    failures = []
    for entry in doc['entries']:
        path = safe_path(root, entry['path'])
        try:
            if entry['kind'] == 'symlink':
                okay = path.is_symlink() and os.readlink(path) == entry['target']
            else:
                okay = path.is_file() and not path.is_symlink()
                if okay:
                    with path.open('rb') as stream:
                        okay = digest(stream) == entry['sha256']
            if not okay:
                failures.append(entry['path'])
        except OSError:
            failures.append(entry['path'])
    if failures:
        raise RuntimeError('Archive verification failed: ' + repr(failures[:20]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['manifest', 'verify', 'prune'])
    parser.add_argument('--archive')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    mp = Path(args.manifest)
    if args.action == 'manifest':
        doc = manifest(args.archive)
        mp.write_text(json.dumps(doc, indent=2) + '\n')
        print('MANIFEST_COMPLETE', len(doc['entries']), flush=True)
        return
    doc = json.loads(mp.read_text())
    root = Path(args.root).resolve(strict=True)
    if root == Path('/') or not (root / 'experiments').is_dir():
        raise ValueError('Expected an existing root containing experiments/')
    with mp.open('rb') as stream:
        checksum = digest(stream)
    if args.action == 'prune':
        receipt = json.loads(Path(args.receipt).read_text())
        if receipt.get('manifest_sha256') != checksum or receipt.get('status') != 'verified':
            raise ValueError('Missing matching destination verification receipt')
    verify(root, doc)
    if args.action == 'verify':
        result = {'status': 'verified', 'root': str(root), 'manifest_sha256': checksum,
                  'verified_entries': len(doc['entries'])}
        Path(args.receipt).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    # All entries matched immediately before deletion. Never follow directory links.
    parents = set()
    for entry in doc['entries']:
        path = safe_path(root, entry['path'])
        path.unlink()
        parents.update(p for p in path.parents if p != root and root in p.parents)
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        if parent == root / 'experiments':
            continue
        try:
            parent.rmdir()  # Empty directories only; retained banks cannot be removed.
        except OSError:
            pass
    print('PRUNED_VERIFIED_ENTRIES', len(doc['entries']), flush=True)


if __name__ == '__main__':
    main()
