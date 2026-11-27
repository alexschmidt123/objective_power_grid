"""Summarize the four terminal checkpoints and paired case-level differences."""
import json
import sys
from pathlib import Path

import numpy as np


def estimate(values):
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return dict(nats=mean, case_mcse=se, case_ci95=[mean - 1.96 * se, mean + 1.96 * se])


def summarize(directory):
    names = ["homogeneous_learned", "homogeneous_uniform", "heterogeneous_learned", "heterogeneous_uniform"]
    output = dict(arms={}, differences={}, scope="Single training seed 101; intervals conditional on training and shared contrast bank.")
    scores = {}
    for index, name in enumerate(names):
        receipt_path = directory / f"{index}.path"
        run = Path(receipt_path.read_text().strip()) if receipt_path.exists() else None
        if run is None or not (run / "canonical_evaluation.npz").is_file():
            output["arms"][name] = dict(status="missing_evaluation", run=str(run) if run else None)
            continue
        with np.load(run / "canonical_evaluation.npz", allow_pickle=False) as data:
            receipt = json.loads(str(data["receipt"]))
            values = data["scores"].copy()
        if values.shape != (2000,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid case scores for {name}")
        manifest = json.loads((run / "manifest.json").read_text())
        expected_family = ("observed_history_moe", "structured_history_moe")[index // 2]
        expected_method = ("moe-sboed", "moe-sboed-uniform")[index % 2]
        if (manifest["architecture"], receipt["method"], receipt["seed"]) != (expected_family, expected_method, 101):
            raise ValueError(f"Wrong policy for {name}")
        protocol = receipt["protocol"]
        if index and "protocol" in output and protocol != output["protocol"]:
            raise ValueError("Paired contrasts require identical scoring protocols")
        output["protocol"] = protocol
        output["arms"][name] = dict(run=str(run), receipt=receipt, **estimate(values))
        scores[index] = values
    contrasts = {"routing_homogeneous": [(0, 1), (1, -1)], "routing_heterogeneous": [(2, 1), (3, -1)],
                 "heterogeneity_learned": [(2, 1), (0, -1)], "heterogeneity_uniform": [(3, 1), (1, -1)],
                 "routing_by_heterogeneity": [(2, 1), (3, -1), (0, -1), (1, 1)]}
    for name, terms in contrasts.items():
        if all(i in scores for i, _ in terms):
            output["differences"][name] = estimate(sum(w * scores[i] for i, w in terms))
    (directory / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
