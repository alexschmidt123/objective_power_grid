"""Allocate fresh experiment directories using the effective config, horizon and methods."""
from datetime import datetime
from itertools import count
from pathlib import Path
import re

TOKEN = r"[A-Za-z0-9][A-Za-z0-9_-]*"
RUN_NAME = re.compile(rf"(?P<stamp>\d{{8}}-\d{{4}})-(?P<experiment>{TOKEN})_T(?P<T>[1-9]\d*)-(?P<methods>{TOKEN}(?:\+{TOKEN})*)")


def allocate_run(parent, experiment, horizon, methods, *, start=None):
    methods = tuple(methods)
    if not methods or not all(re.fullmatch(TOKEN, item) for item in (experiment, *methods)):
        raise ValueError("Experiment and method names must be nonempty filename-safe identifiers")
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    stamp = (start or datetime.now()).strftime("%m%d%Y-%H%M")
    prefix = Path(parent)
    prefix.mkdir(parents=True, exist_ok=True)
    for attempt in count(1):
        suffix = "" if attempt == 1 else f"_r{attempt}"
        path = prefix / f"{stamp}-{experiment}{suffix}_T{horizon}-{'+'.join(methods)}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
