"""Serialize bank writers and publish completed generations without truncation."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import uuid


@contextmanager
def bank_lock(destination: Path):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode outside the directory being published/replaced.
    with (destination.parent / ('.' + destination.name + '.lock')).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def staged_bank(destination: Path):
    """Caller holds bank_lock; failures never publish partial files.

    Previous directories are retained so existing mmap readers keep valid files.
    Replacement has a short rename gap; readers may retry, never see partial data.
    """
    destination = Path(destination).resolve()
    stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.building-',
                                 dir=destination.parent))
    try:
        yield stage
        previous = None
        if destination.exists():
            previous = destination.with_name('.' + destination.name + '.previous-' + uuid.uuid4().hex)
            os.rename(destination, previous)
        try:
            os.rename(stage, destination)
        except BaseException:
            if previous is not None:
                os.rename(previous, destination)
            raise
    except BaseException:
        # Retain failed generation for diagnosis; it is never a usable bank.
        raise


def write_manifest(path: Path, payload: dict):
    target = Path(path) / 'meta' / 'completion.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('completion.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, target)
