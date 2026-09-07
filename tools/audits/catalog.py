"""Reusable dense-bank catalog and candidate-proposal helpers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib, json, math
import numpy as np

@dataclass(frozen=True)
class Catalog:
    designs: tuple[tuple[float, int, float], ...]

    @property
    def buses(self) -> tuple[int, ...]:
        return tuple(sorted({int(row[1]) for row in self.designs}))

    @property
    def durations(self) -> tuple[float, ...]:
        return tuple(sorted({round(float(row[2]), 10) for row in self.designs}))


def load_catalog(bank_dir: Path) -> Catalog:
    path = bank_dir / "meta" / "catalog.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    designs = tuple(
        (float(row[0]), int(row[1]), float(row[2])) for row in raw["designs"]
    )
    if not designs:
        raise ValueError(f"empty action catalog: {path}")
    return Catalog(designs=designs)


def resolve_pool_actions(
    catalog: Catalog,
    durations: Iterable[float],
    *,
    tolerance: float = 5e-7,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    requested = tuple(sorted(set(round(float(x), 10) for x in durations)))
    if not requested:
        raise ValueError('At least one duration is required')
    by_duration: dict[float, np.ndarray] = {}
    for duration in requested:
        ids = np.asarray(
            [
                i
                for i, (_, _, bank_duration) in enumerate(catalog.designs)
                if abs(float(bank_duration) - duration) <= tolerance
            ],
            dtype=np.int64,
        )
        if ids.size != len(catalog.buses):
            raise ValueError(
                f"duration {duration:g}s maps to {ids.size} bank actions; "
                f"expected one action for each of {len(catalog.buses)} buses"
            )
        if sorted(catalog.designs[int(i)][1] for i in ids) != list(catalog.buses):
            raise ValueError(f'duration {duration:g}s does not cover each bus exactly once')
        by_duration[duration] = ids
    pool_ids = np.concatenate([by_duration[d] for d in requested])
    return pool_ids, by_duration


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def candidate_sets(durations, baseline, count, seed):
    if count < 2 or len(baseline) != 6:
        raise ValueError('Require >=2 candidates and a six-duration baseline')
    if count > math.comb(len(durations), 6):
        raise ValueError('Candidate count exceeds available combinations')
    rng = np.random.default_rng(seed)
    rows = {tuple(sorted(baseline)), tuple(map(int, np.linspace(0, len(durations)-1, 6).round()))}
    while len(rows) < count:
        rows.add(tuple(sorted(rng.choice(len(durations), 6, replace=False).tolist())))
    return sorted(rows)


def proxy_scores(centres: np.ndarray, n_durations: int, n_buses: int) -> tuple[np.ndarray, np.ndarray]:
    """Return duration informativeness and pairwise response dissimilarity."""
    shaped = centres.reshape(n_durations, n_buses, centres.shape[1], centres.shape[2])
    # Preserve particle-dependent response shape; remove the per-action mean so
    # duration separation is not merely a waveform offset.
    fingerprints = shaped - shaped.mean(axis=2, keepdims=True)
    fingerprints = fingerprints.reshape(n_durations, -1).astype(np.float64)
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    normalized = fingerprints / np.maximum(norms, 1e-12)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    dissimilarity = 1.0 - similarity
    info = shaped.var(axis=2).mean(axis=(1, 2)).astype(np.float64)
    info = info / max(float(np.max(info)), 1e-12)
    return info, dissimilarity


def proxy_value(key: tuple[int, ...], info: np.ndarray, distance: np.ndarray) -> float:
    ids = np.asarray(key, dtype=np.int64)
    pair = distance[np.ix_(ids, ids)]
    upper = pair[np.triu_indices(len(ids), 1)]
    # Require all six durations to be informative; reward non-redundant shapes.
    return float(0.55 * np.min(info[ids]) + 0.25 * np.mean(info[ids]) + 0.20 * np.mean(upper))
