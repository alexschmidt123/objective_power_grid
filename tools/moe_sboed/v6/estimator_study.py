"""Run the fixed seed-101 routing × estimator comparison on one reserved LabPC GPU."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from src.research.moe_sboed.v6.evaluate import PROTOCOL
from src.research.moe_sboed.v6.train import put_json, sha
from tools.moe_sboed.v6.labpc_seed101 import references, historical_feature_scores

CONFIGS = {name: Path("configs/moe_sboed/v6") / f"{stem}_matched20k_seed101.yaml"
           for name, stem in (("sace", "expert_history_sace"), ("spce", "expert_history"))}
METHODS = ("moe-sboed", "moe-sboed-uniform")
CELLS = list(itertools.product(CONFIGS, METHODS))
TARGET = 13.2


def verify_design():
    protocols = []
    for name, path in CONFIGS.items():
        cfg = yaml.safe_load(path.read_text())
        training = cfg["training"]
        if training.pop("objective") != f"terminal_{name}":
            raise ValueError(f"Unexpected training objective in {path}")
        training.pop("proposal", None)
        protocols.append(cfg)
    if protocols[0] != protocols[1] or protocols[0]["training"]["updates"] != 20000:
        raise ValueError("Configuration mismatch in paired study")


def difference(values):
    n = len(values)
    mean = float(values.mean())
    mcse = float(values.std(ddof=1)/np.sqrt(n))
    return dict(cases=n, mean_nats=mean, case_mcse=mcse, case_ci95=[mean-1.96*mcse, mean+1.96*mcse])


def report(out, refs):
    rows, scores = {}, {}
    lock = json.loads(PROTOCOL.read_text())["evaluation"]
    for estimator, method in CELLS:
        name = f"{estimator}_{method}"
        summary = out/name/"comparison.json"
        if not summary.exists():
            continue
        run = Path(json.loads(summary.read_text())[method]["run_dir"])
        with np.load(run/"canonical_evaluation.npz") as archive:
            info = json.loads(str(archive["receipt"]))
            vector = archive["scores"]
        manifest = json.loads((run/"manifest.json").read_text())
        if (info["protocol"] != lock or info["method"] != method or info["seed"] != 101
                or manifest["config_sha256"] != sha(CONFIGS[estimator])
                or info["checkpoint_sha256"] != sha(run/"checkpoint.pt")
                or vector.shape != (lock["cases"],) or not np.isfinite(vector).all()):
            raise ValueError(f"Evaluation provenance mismatch for {name}")
        matched = info["completed_updates"] == 20000 and info["training_status"] == "complete"
        rows[name] = dict(run=str(run), updates=info["completed_updates"], matched_budget=matched,
                          eig=difference(vector), exceeds_13_2=float(vector.mean())>TARGET,
                          budget=json.loads((run/"budget.json").read_text()),
                          differences={key:difference(vector-ref) for key,ref in refs.items()})
        if matched:
            scores[(estimator,method)] = vector
    comparisons = {}
    for estimator in CONFIGS:
        full, uniform = [(estimator,m) for m in METHODS]
        if full in scores and uniform in scores:
            comparisons[f"{estimator}_full_minus_uniform"] = difference(scores[full]-scores[uniform])
    for method in METHODS:
        if ("sace",method) in scores and ("spce",method) in scores:
            comparisons[f"{method}_sace_minus_spce"] = difference(scores["sace",method]-scores["spce",method])
    if len(scores)==4:
        full,uniform=METHODS
        comparisons["routing_gain_sace_minus_spce"] = difference((scores["sace",full]-scores["sace",uniform]) -
                                                                 (scores["spce",full]-scores["spce",uniform]))
    put_json(out/"results.json",dict(cells=rows,matched_seed101_contrasts=comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    out = args.directory
    out.mkdir(parents=True, exist_ok=True)
    verify_design()
    refs = references(out)
    refs.update(historical_feature_scores())
    report(out, refs)
    if args.preflight:
        return
    state = {}
    for estimator, method in CELLS:
        name = f"{estimator}_{method}"
        directory = out/name
        if (directory/"comparison.json").is_file():
            state[name] = "previously_scored"
            continue
        if directory.exists():
            state[name] = "incomplete_existing_attempt"
            continue
        state[name] = "running"
        put_json(out/"status.json",state)
        cmd = [sys.executable, "-m", "tools.moe_sboed.v6.labpc_seed101", str(directory),
               "--config", str(CONFIGS[estimator]), "--methods", method]
        rc = subprocess.run(cmd).returncode
        state[name] = "scored" if rc==0 else f"failed_exit_{rc}"
        put_json(out/"status.json",state)
        report(out,refs)
    put_json(out/"status.json",state)
    if any(s not in ("previously_scored","scored") for s in state.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
