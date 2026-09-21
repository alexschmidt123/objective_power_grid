"""Explicit, read-only checkpoint interventions for the independent SIR MoE.

All interventions keep the trained encoder and heads frozen. They are not
independently trained expert policies. Full training stays behind run.sh.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import load_config, with_step_number
from src.domains.sir.context import build_sir_context
from src.objectives.eig.independent_moe import batch_rollout
from src.objectives.eig.vector import VectorEIGEngine, _load_policy


class FrozenIntervention(nn.Module):
    def __init__(self, policy, expert=None):
        super().__init__()
        self.policy, self.expert = policy, expert

    def forward(self, *inputs):
        features = self.policy.features(*inputs[:7])
        logits = (torch.stack([head(features) for head in self.policy.experts], 1).mean(1)
                  if self.expert is None else self.policy.experts[self.expert](features))
        return logits.masked_fill(~inputs[-1], -1e9)

    def distribution(self, *inputs):
        return torch.distributions.Categorical(logits=self(*inputs))


@torch.no_grad()
def evaluate(source, output, eval_seeds):
    saved = json.loads((source / "run_config.json").read_text())
    cfg = with_step_number(load_config(Path(saved["source_config"])), int(saved["T"]))
    cfg.raw["observation"]["noise_sigma"] = float(saved["noise_sigma"])
    ctx = build_sir_context(cfg, out_dir=source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _load_policy(ctx, "moe_sboed", device)
    assert policy.routing == "learned" and policy.n_experts == 4
    engine = VectorEIGEngine(ctx, device)
    particles = torch.tensor(ctx.particle_features, dtype=torch.float32, device=device)[None]
    particles[..., 2] = 0
    val = torch.tensor(np.stack([s["obs_clean"] for s in ctx.validation_systems[:256]]), dtype=torch.float32, device=device)
    test = torch.tensor(np.stack([s["obs_clean"] for s in ctx.test_systems[:512]]), dtype=torch.float32, device=device)
    policies = {"MoE-sBOED": policy, "Uniform weights (frozen MoE)": FrozenIntervention(policy)}
    policies.update({f"Expert {e + 1} alone": FrozenIntervention(policy, e) for e in range(4)})
    val_scores = {}
    # Select a single expert on validation, before evaluating its test scores.
    for name, candidate in policies.items():
        *_, gains, _ = batch_rollout(ctx, candidate, val, engine.centres, particles,
                                    seed=int(saved["seed"]), stochastic=False,
                                    rollout_ids=np.arange(len(val)) + 70000)
        val_scores[name] = float(gains.double().mean())
    best_expert = max((name for name in policies if name.startswith("Expert")), key=lambda name: val_scores[name])
    records, summaries, state_diagnostics = [], [], []
    for seed in eval_seeds:
        existing = json.loads((source / "evaluations" / f"seed_{seed}" / "eval/vector_eig_results.json").read_text())
        for name, candidate in policies.items():
            states, _, _, _, gains, actions = batch_rollout(
                ctx, candidate, test, engine.centres, particles, seed=seed,
                stochastic=False, rollout_ids=np.arange(len(test)))
            assert torch.isfinite(gains).all() and (actions[:, 1:] > actions[:, :-1]).all()
            if name == "MoE-sBOED":
                original = sorted(existing["rollouts"], key=lambda row: row["theta_id"])
                assert actions.cpu().tolist() == [row["sequence"] for row in original], "Saved policy action mismatch"
                np.testing.assert_allclose(gains.cpu().numpy(), [row["terminal_eig"] for row in original], atol=1e-5, rtol=0)
                for stage, inputs in enumerate(states):
                    features = policy.features(*inputs[:7])
                    choices = torch.stack([head(features).masked_fill(~inputs[-1], -1e9).argmax(-1)
                                           for head in policy.experts], 1)
                    disagreement = torch.stack([(choices[:, i] != choices[:, j]).float()
                                                for i in range(4) for j in range(i + 1, 4)]).mean()
                    state_diagnostics.append({"stage": stage, "eval_seed": seed,
                                              "pairwise_expert_action_disagreement": float(disagreement)})
            summaries.append({"T": ctx.horizon, "eval_seed": seed, "method": name,
                              "mean_eig": float(gains.double().mean()),
                              "unique_sequences": len(torch.unique(actions, dim=0))})
            for i, (score, sequence) in enumerate(zip(gains.cpu().tolist(), actions.cpu().tolist())):
                records.append({"method": name, "eval_seed": seed, "theta_id": i,
                                "eig": score, "sequence": sequence})
    checkpoint = source / "model/moe_sboed.pth"
    report = {"T": ctx.horizon, "source": str(source), "training_seed": int(saved["seed"]),
              "evaluation_seeds": eval_seeds, "n_test_systems": len(test),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "validation_scores": val_scores, "best_expert_by_validation": best_expert,
              "summaries": summaries, "records": records, "state_diagnostics": state_diagnostics,
              "original_evaluation_reproduced": True,
              "interpretation": "Frozen interventions measure reliance on routing; forced experts see trajectories outside their usual routing distribution."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "expert_interventions.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "state_diagnostics"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-seeds", default="1001,1002,1003,1004,1005")
    args = parser.parse_args()
    evaluate(args.source, args.output, [int(s) for s in args.eval_seeds.split(",")])
