"""Paired terminal comparisons for a fixed two-arm contrast-count study."""
import json
import sys
from pathlib import Path

import numpy as np


def read(directory, bank):
    with np.load(directory / f"{bank}_evaluation.npz", allow_pickle=False) as data:
        scores = data["scores"].copy()
        receipt = json.loads(str(data["receipt"]))
    if scores.shape != (2000,) or not np.isfinite(scores).all():
        raise ValueError("Invalid saved terminal scores")
    return scores, receipt


def estimate(x):
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return dict(nats=mean, case_mcse=se, conditional_ci95=[mean-1.96*se, mean+1.96*se])


def report(output, previous):
    results = {"missing": [], "cases": 2000, "training_seed": 101}
    for bank in ("development", "canonical"):
        section = results[bank] = {}
        values = {}
        protocol = None
        for i, method in enumerate(("moe-sboed", "moe-sboed-uniform")):
            for k in (2000, 8000):
                key = f"{method}_K{k}"
                pathfile = (previous if k == 2000 else output) / f"{i}.path"
                if not pathfile.is_file():
                    results["missing"].append([bank, key])
                    continue
                directory = Path(pathfile.read_text().strip())
                location = output / f"reference-{i}" if bank == "development" and k == 2000 else directory
                if not (location / f"{bank}_evaluation.npz").is_file():
                    results["missing"].append([bank, key])
                    continue
                x, receipt = read(location, bank)
                manifest = json.loads((directory / "manifest.json").read_text())
                budget = json.loads((directory / "budget.json").read_text())
                if (manifest["train_contrasts"], receipt["method"], receipt["seed"]) != (k, method, 101):
                    raise ValueError("Arm mismatch")
                if protocol is not None and receipt["protocol"] != protocol:
                    raise ValueError("Protocol mismatch")
                protocol = receipt["protocol"]
                values[method,k] = x
                section[key] = dict(**estimate(x), run=str(directory), receipt=receipt, budget=budget)
            if all((method,k) in values for k in (2000,8000)):
                section[method+"_gain"] = estimate(values[method,8000]-values[method,2000])
        if all((m,8000) in values for m in ("moe-sboed","moe-sboed-uniform")):
            section["learned_minus_uniform_K8000"] = estimate(values["moe-sboed",8000]-values["moe-sboed-uniform",8000])
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    report(*map(Path, sys.argv[1:]))
