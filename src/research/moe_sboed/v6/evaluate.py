"""Score terminal location policies with the archived full-horizon V4/V5 protocol."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.objective import trajectory
from src.research.moe_sboed.v6.policy import build_policy, METHODS
from src.research.moe_sboed.v6.train import sha

ARCHIVE = Path("experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109")
PROTOCOL = Path("reports/moe_sboed_v4_allmethods_20261117/PROTOCOL.json")


def require_originals(directory, lock):
    files = {
        "scorer_sha256": ARCHIVE / "source/frozen_evaluator.py",
        "bank_sha256": directory / "development_validation_bank.npz",
        "uniforms_sha256": directory / "routing_uniforms.npy",
    }
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Original evaluation input required: {path}")
        if sha(path) != lock[key]:
            raise ValueError(f"Frozen protocol hash mismatch: {path}")
    return files


@torch.inference_mode()
def score_cases(model, bench, truth, noise, contrasts, scorer, chunk):
    values = []
    for start in range(0, len(truth), 16):
        theta = truth[start:start + 16]
        actions, obs = trajectory(bench, model, theta, noise[start:start + 16])
        result = dict(truth=theta, actions=actions.unbind(1), observations=obs.unbind(1))
        curve, _ = scorer(bench, result, contrasts, chunk, theta.device)
        values.append(curve[:, -1].numpy())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe-sboed-v6", action="store_true", required=True)
    parser.add_argument("--evaluate", action="store_true", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=ARCHIVE / "evaluation")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--allow-time-limited", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    lock = json.loads(PROTOCOL.read_text())["evaluation"]
    for key in ("cases", "contrasts", "bank_sha256", "scorer_sha256", "uniforms_sha256"):
        if cfg["evaluation"][key] != lock[key]:
            raise ValueError(f"V6 evaluation conflicts with the baseline protocol for {key}")
    if (cfg["horizon"], cfg["observation"]["noise_sigma"], cfg["location_finding"]["design_bounds"]) != (
        lock["horizon"], lock["sigma"], [-4, 4]
    ):
        raise ValueError("Canonical horizon, noise and design bounds required")
    checkpoint = args.run_dir / "checkpoint.pt"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    manifest = state["manifest"]
    target = cfg["training"]["updates"]
    if (manifest["method"], manifest["config_sha256"], manifest["seed"], manifest["smoke"],
        manifest["target_updates"]) != (args.method, sha(args.config), cfg["seed"], False, target):
        raise ValueError("Checkpoint does not match the requested training configuration")
    status = json.loads((args.run_dir / "status.json").read_text())
    budget = json.loads((args.run_dir / "budget.json").read_text())
    update = state["update"]
    if update != status["completed_updates"] or update != budget["updates"] or status["target_updates"] != target:
        raise ValueError("Inconsistent checkpoint, training status and budget")
    complete = status["status"] == "complete" and update == target
    capped = args.allow_time_limited and status["status"] == "time_limited" and 0 < update < target
    if not (complete or capped):
        raise ValueError("Expected a terminal checkpoint under the requested budget")
    if args.development:
        bank = args.bank_dir / "development_validation_bank.npz"
        scorer = ARCHIVE / "source/frozen_evaluator.py"
        if sha(bank) != cfg["development"]["bank_sha256"] or sha(scorer) != lock["scorer_sha256"]:
            raise ValueError("Development bank or frozen scorer digest mismatch")
        files = {"bank_sha256": bank, "scorer_sha256": scorer}
        lock = {k:v for k,v in lock.items() if k != "uniforms_sha256"}
        lock.update(bank_sha256=cfg["development"]["bank_sha256"], purpose="development",
                    generation_seed=cfg["development"]["seed"])
    else:
        files = require_originals(args.bank_dir, lock)
    model = build_policy(args.method, cfg, LocationFinding(cfg)).eval()
    model.load_state_dict(state["state_dict"])
    if args.preflight:
        print(f"Verified checkpoint and frozen inputs for {args.method}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Run scoring on an allocated GPU")
    spec = importlib.util.spec_from_file_location("v2_frozen_evaluator", files["scorer_sha256"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(files["bank_sha256"], allow_pickle=False) as bank:
        truth, noise, contrasts = [torch.as_tensor(bank[k], device="cuda", dtype=torch.float64)
                                  for k in ("truth", "noise", "contrasts")]
    values = score_cases(model.cuda(), LocationFinding(cfg), truth, noise, contrasts,
                         module.contrast_scores, lock["chunk"])
    ceiling = math.log1p(lock["contrasts"])
    if values.shape != (lock["cases"],) or not np.isfinite(values).all() or np.any(values > ceiling + 1e-9):
        raise ValueError("Invalid full-horizon score vector")
    receipt = dict(method=args.method, seed=cfg["seed"], protocol=lock, ceiling_nats=ceiling,
                   training_status=status["status"], completed_updates=update, target_updates=target,
                   checkpoint_sha256=sha(checkpoint), eig_nats=values.mean().item(),
                   case_mcse_nats=values.std(ddof=1).item() / math.sqrt(len(values)))
    destination = args.run_dir / ("development_evaluation.npz" if args.development else "canonical_evaluation.npz")
    with tempfile.NamedTemporaryFile(dir=args.run_dir, suffix=".npz") as temp:
        np.savez(temp, scores=values, receipt=json.dumps(receipt))
        temp.flush()
        os.link(temp.name, destination)
    print(json.dumps(receipt), flush=True)
    if cfg.get("architecture") in ("belief_expert_history", "observed_history_moe", "structured_history_moe", "relative_history_moe"):
        from tools.moe_sboed.v6.belief_timing import record_design_timing
        record_design_timing(args.run_dir, model, LocationFinding(cfg), cfg["horizon"])


if __name__ == "__main__":
    main()
