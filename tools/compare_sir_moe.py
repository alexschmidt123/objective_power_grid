"""Compare fresh independent SIR MoE with explicit historical baseline runs.

Requires matching saved experiment settings, training/evaluation seeds, and
system counts. Never selects a baseline run by its performance.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = {"dad_eig": "DAD", "rl_sboed_eig": "RL-sBOED", "step_dad": "Step-DAD",
           "myopic_delta_h": "Myopic", "fixed_open_loop": "Fixed", "random": "Random",
           "moe_sboed": "MoE-sBOED"}


def read_json(path):
    return json.loads(path.read_text())


def compare(campaign, references):
    records, provenance, mechanisms = [], [], []
    for horizon, reference in sorted(references.items()):
        candidates = list(campaign.glob(f"*_EIG_T{horizon}_Nobs1_sigma1"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one new T={horizon} run: {candidates}")
        run = candidates[0]
        manifest = read_json(run / "evaluation_manifest.json")
        assert manifest["training_seed"] == 101
        assert manifest["evaluation_seeds"] == [1001, 1002, 1003, 1004, 1005]
        for seed in manifest["evaluation_seeds"]:
            new_dir = run / "evaluations" / f"seed_{seed}"
            old_dir = reference / "evaluations" / f"seed_{seed}"
            new, old = read_json(new_dir / "run_config.json"), read_json(old_dir / "run_config.json")
            for key in ("T", "N_obs", "noise_sigma", "seed", "sir_ode", "prior", "data_generation"):
                if new.get(key) != old.get(key):
                    raise ValueError(f"T={horizon} seed={seed} mismatch {key}: {new.get(key)} vs {old.get(key)}")
            for key in ("eig_epochs", "eig_steps_per_epoch", "batch_size", "learning_rate", "eig_validation_systems"):
                assert new["training"][key] == old["training"][key], key
            for source, directory in (("new", new_dir), ("historical", old_dir)):
                path = directory / "eval/terminal_eig_summary.csv"
                rows = list(csv.DictReader(path.open()))
                expected = {"moe_sboed"} if source == "new" else set(METHODS) - {"moe_sboed"}
                assert {r["method"] for r in rows} == expected, (path, rows)
                for row in rows:
                    assert int(row["n"]) == 512 and int(row["eval_seed"]) == seed
                    value = float(row["terminal_eig_mean"])
                    assert math.isfinite(value)
                    records.append({"T": horizon, "training_seed": 101, "eval_seed": seed,
                                    "method": METHODS[row["method"]], "eig": value,
                                    "unique_sequences": int(row["n_unique_sequences"]),
                                    "source": str(path)})
                provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            results = read_json(new_dir / "eval/vector_eig_results.json")
            for row in results["rollouts"]:
                assert all(a < b for a, b in zip(row["sequence"], row["sequence"][1:])), row
                assert len(row["router_trace"]) == horizon
            for stage in range(horizon):
                selected, weights = [0] * 4, [0.] * 4
                routes = []
                for row in results["rollouts"]:
                    trace = row["router_trace"][stage]
                    experts = trace["selected_experts"][0]
                    assert len(experts) == len(set(experts)) == 2
                    routes.append(tuple(sorted(experts)))
                    for e, w in zip(experts, trace["selected_weights"][0]):
                        selected[e] += 1
                        weights[e] += w
                count = len(results["rollouts"])
                mechanisms.append({"T": horizon, "eval_seed": seed, "stage": stage,
                                   "selected_fraction": [v / count for v in selected],
                                   "mean_routing_weight": [v / count for v in weights],
                                   "distinct_expert_pairs": len(set(routes))})
    summary = []
    for method in METHODS.values():
        for horizon in sorted(references):
            values = [r["eig"] for r in records if r["T"] == horizon and r["method"] == method]
            assert len(values) == 5
            summary.append({"method": method, "T": horizon, "mean_eig": statistics.mean(values),
                            "eval_seed_sd": statistics.stdev(values), "training_seeds": 1, "evaluation_seeds": 5})
    out = campaign / "comparison"
    out.mkdir(exist_ok=True)
    for name, rows in (("summary.csv", summary), ("per_seed.csv", records)):
        with (out / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    doc = {"metric": "finite-particle posterior entropy reduction (nats)",
           "training_seed": 101, "evaluation_seeds": [1001,1002,1003,1004,1005],
           "n_test_systems": 512, "noise_sigma": 1.0, "summary": summary,
           "routing": mechanisms, "sources": provenance,
           "limitations": ["One training seed; evaluation-seed SD is not training-seed uncertainty.",
                           "Historical bank identity is checked through saved generation settings, not historical content hashes.",
                           "MoE architecture and optimization differ; this is not a parameter-matched ablation."]}
    (out / "report.json").write_text(json.dumps(doc, indent=2) + "\n")
    lines = ["# Independent MoE-sBOED SIR ODE test", "", "Mean EIG (nats), train seed 101, five evaluation seeds, 512 systems per seed.", "",
             "| Method | T=3 | T=4 | T=5 |", "|---|---:|---:|---:|"]
    for method in METHODS.values():
        vals = [next(r["mean_eig"] for r in summary if r["method"] == method and r["T"] == t) for t in (3,4,5)]
        lines.append("| " + method + " | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines += ["", "MoE: four experts, learned top-2 sparse dispatch at every stage, fresh PPO training, no teacher or baseline checkpoint, validation-EIG checkpoint selection.", "",
              "This is one training seed. Routing statistics and all source paths/checksums are in report.json; evaluation-seed variation is reported separately in summary.csv."]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--reference", action="append", required=True, help="T=path (one per horizon)")
    args = parser.parse_args()
    compare(args.campaign, {int(s.split("=", 1)[0]): Path(s.split("=", 1)[1]) for s in args.reference})
