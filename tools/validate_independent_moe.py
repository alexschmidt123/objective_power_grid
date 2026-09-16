"""Read-only verification of batched and scalar SIR MoE execution."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.config import load_config, with_step_number
from src.domains.sir.context import build_sir_context
from src.objectives.eig.independent_moe import batch_rollout
from src.objectives.eig.vector import VectorEIGEngine, _load_policy, _rollout


def validate(run):
    saved = json.loads((run / "run_config.json").read_text())
    cfg = with_step_number(load_config(Path(saved["source_config"])), int(saved["T"]))
    ctx = build_sir_context(cfg, out_dir=run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _load_policy(ctx, "moe_sboed", device)
    engine = VectorEIGEngine(ctx, device)
    systems = ctx.test_systems[:8]
    clean = torch.tensor(np.stack([s["obs_clean"] for s in systems]), device=device, dtype=torch.float32)
    particles = torch.tensor(ctx.particle_features, device=device, dtype=torch.float32)[None]
    particles[..., 2] = 0
    *_, gains, actions = batch_rollout(ctx, policy, clean, engine.centres, particles,
                                      seed=1001, stochastic=False, rollout_ids=np.arange(8))
    with patch.object(VectorEIGEngine, "action_scores", side_effect=AssertionError("planner invoked")), \
         patch.object(VectorEIGEngine, "two_step_scores", side_effect=AssertionError("planner invoked")), \
         torch.no_grad():
        rows = [_rollout(ctx, engine, s, rollout_id=i, method="moe_sboed", dad=policy,
                         fixed_sequence=[], n_fantasies=4, eval_seed=1001) for i, s in enumerate(systems)]
    assert actions.cpu().tolist() == [r["sequence"] for r in rows], "batched/scalar actions differ"
    error = np.max(np.abs(gains.cpu().numpy() - [r["terminal_eig"] for r in rows]))
    assert error < 1e-5, error
    assert all(len(r["router_trace"]) == ctx.horizon for r in rows)
    return {"run": str(run), "T": ctx.horizon, "systems": 8, "max_eig_error": float(error),
            "identical_actions": True, "planner_calls": 0, "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [validate(run) for run in args.runs]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
