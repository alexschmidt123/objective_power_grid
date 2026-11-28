"""Paired development-only router interventions for the homogeneous location pilot."""
import argparse
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.evaluate import ARCHIVE
from src.research.moe_sboed.v6.objective import trajectory
from src.research.moe_sboed.v6.policy import build_policy
from src.research.moe_sboed.v6.train import put_json, sha
from src.results.run_directory import allocate_run


def rollout_with_gates(model, bench, truth, noise, replacements=None):
    recorded = []

    def gate_hook(module, inputs, logits):
        step = len(recorded)
        recorded.append(logits.softmax(-1))
        if replacements is not None:
            return replacements[:, step].log()
    handle = model.router.register_forward_hook(gate_hook)
    try:
        actions, observations = trajectory(bench, model, truth, noise)
    finally:
        handle.remove()
    return actions, observations, torch.stack(recorded, dim=1)


def terminal_scores(scorer, bench, truth, actions, obs, contrasts):
    values = []
    for i in range(0, len(truth), 16):
        sl = slice(i, i + 16)
        case = dict(truth=truth[sl], actions=actions[sl].unbind(1), observations=obs[sl].unbind(1))
        curve, _ = scorer(bench, case, contrasts, 16384, truth.device)
        values.append(curve[:, -1].numpy())
    values = np.concatenate(values)
    if not np.isfinite(values).all():
        raise ValueError("Invalid terminal scores")
    return values


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--moe-sboed-v6", "--routing-diagnostic"):
        parser.add_argument(flag, required=True, action="store_true")
    for flag in ("--config", "--run-dir", "--bank-dir", "--experiments"):
        parser.add_argument(flag, required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    ckpt = args.run_dir / "checkpoint.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    manifest = state["manifest"]
    if (manifest["architecture"], manifest["method"], manifest["smoke"], manifest["config_sha256"],
        state["update"]) != ("observed_history_moe", "moe-sboed", False, sha(args.config), cfg["training"]["updates"]):
        raise ValueError("Expected the matching completed learned-router checkpoint")
    bank_path = args.bank_dir / "development_validation_bank.npz"
    scorer_path = ARCHIVE / "source/frozen_evaluator.py"
    if sha(bank_path) != cfg["development"]["bank_sha256"] or sha(scorer_path) != cfg["evaluation"]["scorer_sha256"]:
        raise ValueError("Development bank or scorer mismatch")
    spec = importlib.util.spec_from_file_location("frozen", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    with np.load(bank_path, allow_pickle=False) as data:
        truth, noise, contrasts = [torch.as_tensor(data[k], device="cuda", dtype=torch.float64)
                                   for k in ("truth", "noise", "contrasts")]
    assert len(truth) == 2000 and len(contrasts) == 1000000
    bench = LocationFinding(cfg)
    policy = build_policy("moe-sboed", cfg, bench).cuda().eval()
    policy.load_state_dict(state["state_dict"])
    out = allocate_run(args.experiments, args.config.stem + "_routing_diagnostic", cfg["horizon"], ["moe-sboed"])
    print("OUTPUT", out, flush=True)
    (out / "config_snapshot.yaml").write_bytes(args.config.read_bytes())
    # Donor and recipient histories use disjoint truths and observation-noise rows.
    _, _, donors = rollout_with_gates(policy, bench, truth[:1000], noise[:1000])
    mean = donors.mean(0, keepdim=True).expand_as(donors)
    rng = torch.Generator().manual_seed(42)
    shuffled = torch.stack([donors[torch.randperm(len(donors), generator=rng), t]
                            for t in range(cfg["horizon"])], dim=1)
    truth, noise = truth[1000:], noise[1000:]
    all_scores = {}
    details = dict(checkpoint_sha256=sha(ckpt), bank_sha256=sha(bank_path), scorer_sha256=sha(scorer_path),
                   git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                   calibration_rows=[0,1000], test_rows=[1000,2000], contrasts=len(contrasts),
                   shuffle_seed=42, training_updates=state["update"], mean_eig={}, paired_normal_minus={})
    for name, forced in (("normal", None), ("stage_mean", mean), ("stage_shuffled", shuffled)):
        actions, obs, gates = rollout_with_gates(policy, bench, truth, noise, forced)
        all_scores[name] = terminal_scores(scorer.contrast_scores, bench, truth, actions, obs, contrasts)
        details["mean_eig"][name] = float(all_scores[name].mean())
        if name == "normal":
            details["mean_gate"] = gates.mean((0,1)).tolist()
            details["frac_abs_coordinate_above_3p9"] = (actions.abs() > 3.9).double().mean().item()
            details["terminal_positive_mass"] = {}
            for k in (2000, 8000):
                s = terminal_scores(scorer.contrast_scores, bench, truth, actions, obs, contrasts[:k])
                mass = np.exp(s - math.log1p(k))
                details["terminal_positive_mass"][k] = dict(mean=float(mass.mean()),
                                                            above_point99=float((mass > .99).mean()))
        else:
            delta = all_scores["normal"] - all_scores[name]
            se = delta.std(ddof=1) / math.sqrt(len(delta))
            details["paired_normal_minus"][name] = dict(mean=delta.mean(), case_mcse=se,
                                                        case_ci95=(delta.mean()-1.96*se,delta.mean()+1.96*se))
    np.savez(out / "development_scores.npz", **all_scores)
    put_json(out / "diagnostic.json", details)
    print(json.dumps(details), flush=True)


if __name__ == "__main__":
    main()
