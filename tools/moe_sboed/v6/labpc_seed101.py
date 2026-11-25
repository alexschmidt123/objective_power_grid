"""Sequential seed-101 campaign. Execute at project root with the project Python."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.research.moe_sboed.v6.evaluate import ARCHIVE, METHODS, PROTOCOL, require_originals
from src.research.moe_sboed.v6.train import put_json, sha


def references(out):
    protocol = json.loads(PROTOCOL.read_text())
    inputs = require_originals(ARCHIVE / "evaluation", protocol["evaluation"])
    replay = ARCHIVE.parent / "hec_canonical_evaluation_20261114"
    original = {"V2": ARCHIVE / "evaluation/candidate_scores.npy",
                "V2_uniform": ARCHIVE.parent / "moe_hec_v2_internal_ablation/uniform_scores.npy"}
    vectors, controls = {}, {}
    for name, old in protocol["prior_historical_reproduction"].items():
        p = original.get(name, ARCHIVE / "evaluation" / f"{name}_scores.npy")
        values = np.load(p)
        repeated = np.load(replay / f"{name}_scores.npy")
        diff = float(np.max(np.abs(values - repeated)))
        if (values.shape != (2000,) or repeated.shape != values.shape or
                sha(p) != old["saved_scores_sha256"] or not diff <= 1e-9):
            raise ValueError(f"Historical replay mismatch for {name}")
        controls[name] = dict(saved=str(p), sha256=sha(p), replay_max_abs_error=diff)
        if name not in original:
            vectors[name] = values
    put_json(out / "controls.json", dict(inputs={k: sha(p) for k, p in inputs.items()},
             historical_checkpoint_hashes=json.loads((replay / "CHECKPOINTS.json").read_text()), controls=controls))
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    refs = references(out)
    if args.check:
        return
    cfg = "configs/moe_sboed/v6/location_seed101.yaml"
    rows = {}
    for method in METHODS:
        base = ["bash", "run.sh", "--config", cfg, "--moe-sboed-v6", "--method", method]
        # Exclusive log creation prevents a repeated submission from duplicating training.
        with (out / f"{method}.log").open("x", buffering=1) as log:
            run = None
            with subprocess.Popen(base + ["--seed", "101"], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as proc:
                for line in proc.stdout:
                    log.write(line)
                    if line.startswith("OUTPUT "):
                        run = Path(line.split(maxsplit=1)[1].strip())
                rc = proc.wait()
            if rc or run is None:
                raise RuntimeError(f"{method} training failed; inspect {log.name}")
            subprocess.run(base + ["--evaluate", "--allow-time-limited", "--run-dir", str(run)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(run / "canonical_evaluation.npz") as data:
            values = data["scores"]
            row = json.loads(str(data["receipt"]))
        paired = {}
        for name, baseline in refs.items():
            delta = values - baseline
            mean = float(delta.mean())
            se = float(delta.std(ddof=1) / math.sqrt(len(delta)))
            paired[name] = dict(reference_eig_nats=float(baseline.mean()), difference_nats=mean,
                                case_mcse=se, paired_case_ci95=[mean - 1.96*se, mean + 1.96*se])
        row.update(run_dir=str(run), paired=paired)
        rows[method] = row
        put_json(out / "comparison.json", rows)
        refs[method] = values
        print(method, row["completed_updates"], row["eig_nats"], flush=True)


if __name__ == "__main__":
    main()
