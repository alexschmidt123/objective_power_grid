"""CPU-only paired measurement of training estimators on an already frozen policy."""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.adaptive_contrast import adaptive_scores
from src.research.moe_sboed.v6.objective import scores
from src.research.moe_sboed.v6.policy import build_policy
from src.research.moe_sboed.v6.train import put_json, sha


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    frozen = json.loads(Path("reports/v6_location_seed101_labpc_20261125/comparison.json").read_text())["moe-sboed"]
    run = Path(frozen["run_dir"])
    checkpoint = run / "checkpoint.pt"
    if sha(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("Reference V6 weights have changed")
    cfg = yaml.safe_load((run / "config_snapshot.yaml").read_text())
    experiment = yaml.safe_load(Path("configs/moe_sboed/v6/expert_history_sace_matched20k_seed101.yaml").read_text())
    proposal = experiment["training"]["proposal"]
    bench = LocationFinding(cfg)
    model = build_policy("moe-sboed", cfg, bench).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    seed, cases, contrasts = 2026112521, 256, experiment["training"]["contrasts"]
    rng = torch.Generator().manual_seed(seed)
    outputs = []
    for _ in range(cases // 8):
        truth, noise, base = [torch.randn(shape, dtype=torch.float64, generator=rng)
                              for shape in ((8,4), (8,30), (8,contrasts,4))]
        mask = torch.rand(8, contrasts, 2, generator=rng) < proposal["prior_fraction"]
        prior = scores(bench, model, truth, base, noise)
        adaptive = adaptive_scores(bench, model, truth, base, noise, mask, **proposal)
        outputs.append(torch.stack((prior, adaptive), dim=1))
    prior, adaptive = torch.cat(outputs).unbind(1)
    delta = adaptive - prior
    se = float(delta.std()/math.sqrt(cases))
    gap = float(delta.mean())
    result = dict(seed=seed, cases=cases, contrasts=contrasts, checkpoint_sha256=sha(checkpoint),
                  purpose="Training-estimator diagnostic using a fixed old V6 policy and fresh development cases",
                  proposal=proposal, spce_mean=float(prior.mean()), sace_mean=float(adaptive.mean()),
                  paired_delta=gap, paired_case_ci95=[gap-1.96*se, gap+1.96*se])
    put_json(args.directory/"estimator_probe.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
