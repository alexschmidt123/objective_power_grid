"""Config-driven independent location V6 training on a single allocated GPU."""
import time
START = time.monotonic()

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess

import torch
import yaml

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.objective import cumulative_likelihood, scores
from src.research.moe_sboed.v6.policy import build_policy, METHODS
from src.results.run_directory import allocate_run


def put_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(args, cfg, out):
    if cfg["training"]["objective"] == "vsoed_gmm_i":
        from src.research.moe_sboed.v6.variational import train_native
        return train_native(args, cfg, out, START)
    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError("Training requires a Slurm GPU allocation with CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = cfg["training"]
    horizon, seed = cfg["horizon"], cfg["seed"]
    if settings["initialization"] != "from_scratch" or settings["objective"] not in ("terminal_spce", "terminal_sace"):
        raise ValueError("Expected fresh full-horizon contrastive training")
    batch, micro, contrasts = (settings[k] for k in ("batch_size", "microbatch_size", "contrasts"))
    if batch % micro:
        raise ValueError("Effective batch must be divisible by microbatch size")
    if args.smoke:
        batch, micro, contrasts = 8, 4, 16
    target = 2 if args.smoke else settings["updates"]
    bench = LocationFinding(cfg)
    torch.manual_seed(seed)
    model = build_policy(args.method, cfg, bench).to(device)
    joint_provider = cfg.get("architecture") == "fresh_history_query"
    if joint_provider:
        from src.research.moe_sboed.v6.history_query import training_prefix, update_posterior
        from src.research.moe_sboed.v6.objective import trajectory
        from src.research.engine import logmeanexp
        provider_optimizer = torch.optim.Adam(model.posterior.parameters(), lr=cfg["provider"]["lr"])
        prefix_rng = torch.Generator(device=device).manual_seed(cfg["provider"]["prefix_seed"])
    actor_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(actor_params, lr=settings["learning_rate"], betas=tuple(settings["adam_betas"]))
    rng = torch.Generator(device=device).manual_seed(seed)
    adaptive = settings["objective"] == "terminal_sace"
    if adaptive:
        from src.research.moe_sboed.v6.adaptive_contrast import adaptive_scores
        proposal = settings["proposal"]
        proposal_rng = torch.Generator(device=device).manual_seed(seed + 1000003)
    props = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    manifest = dict(method=args.method, implementation="moe_sboed_v6", seed=seed,
                    horizon=horizon, experts=len(model.experts), initialization="from_scratch",
                    architecture=cfg.get("architecture", "residual_sum"), parameters=sum(p.numel() for p in model.parameters()), train_contrasts=contrasts,
                    effective_batch=batch, microbatch=micro, target_updates=target,
                    objective=settings["objective"], device=str(device), dtype="float64", config_sha256=sha(args.config), smoke=args.smoke,
                    git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                    gpu=props.name if props else None, gpu_memory_bytes=props.total_memory if props else None, job_id=os.getenv("SLURM_JOB_ID"),
                    array_task=os.getenv("SLURM_ARRAY_TASK_ID"), node=os.uname().nodename)
    if joint_provider:
        manifest.update(dtype="float32 policy and posterior; float64 physical trajectories and sPCE",
                        loaded_checkpoints=[], provider_warmup_updates=0,
                        posterior_training="one simulator-labeled prefix per on-policy episode per update")
    put_json(out/"manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    likelihood = cumulative_likelihood if args.smoke else torch.compile(cumulative_likelihood, fullgraph=True)
    update = 0
    terminal_loss = None
    update_seconds = 0.
    deadline = START + settings["limit_seconds"]
    with (out/"train_metrics.jsonl").open("a", buffering=1) as log:
        while update < target and time.monotonic() + max(60, 1.5 * update_seconds) < deadline:
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            if joint_provider:
                prefixes, labels = [], []
            for _ in range(batch // micro):
                truth = torch.randn(micro, 4, generator=rng, dtype=torch.float64, device=device)
                negatives = torch.randn(micro, contrasts, 4, generator=rng, dtype=torch.float64, device=device)
                noise = torch.randn(micro, horizon, generator=rng, dtype=torch.float64, device=device)
                if joint_provider:
                    actions, obs = trajectory(bench, model, truth, noise)
                    ll = likelihood(bench, torch.cat((truth[:,None], negatives),1), actions, obs)
                    values = ll[:,0] - logmeanexp(ll)
                    prefixes.append(training_prefix(actions, obs, prefix_rng))
                    labels.append(truth.float())
                elif adaptive:
                    choose_prior = torch.rand(micro, contrasts, 2, device=device, generator=proposal_rng) < proposal["prior_fraction"]
                    values = adaptive_scores(bench, model, truth, negatives, noise, choose_prior,
                                             proposal["prior_fraction"], proposal["min_scale"], likelihood)
                else:
                    values = scores(bench, model, truth, negatives, noise, likelihood)
                loss = -values.sum() / batch
                loss.backward()
                loss_sum += loss.detach()
            terminal_loss = loss_sum.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
            if not math.isfinite(terminal_loss):
                raise FloatingPointError("Nonfinite terminal contrastive bound")
            optimizer.step()
            if joint_provider:
                provider_nll = update_posterior(model.posterior, provider_optimizer, torch.cat(prefixes), torch.cat(labels))
            update += 1
            if update % settings["decay_every"] == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= settings["lr_decay"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.monotonic() - tick
            if update <= 2 or update % 25 == 0:
                row = dict(update=update, gradient_norm=norm.item(),
                           update_seconds=update_seconds, elapsed_seconds=time.monotonic()-START)
                if joint_provider:
                    row["provider_nll"] = provider_nll
                row["sace" if adaptive else "spce"] = -terminal_loss
                print(json.dumps(row), file=log)
                if update <= 2 or update % 250 == 0:
                    print(json.dumps(row), flush=True)
            if update == 1 or update % settings["checkpoint_every"] == 0:
                save_model(out, model, manifest, update)
                put_json(out/"status.json", dict(status="running", completed_updates=update))
    if update:
        save_model(out, model, manifest, update)
    status = "smoke_complete" if args.smoke and update==target else "complete" if update==target else "time_limited"
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    provider_budget = dict(provider_updates=update, provider_labeled_prefixes=update*batch,
                           provider_pretraining_episodes=0) if joint_provider else {}
    put_json(out/"budget.json", dict(**provider_budget, updates=update, fresh_training_episodes=update*batch,
             simulated_observations=update*batch*horizon, contrast_stage_likelihoods=update*batch*horizon*contrasts,
             inherited_episodes=0, elapsed_seconds=time.monotonic()-START, cpu_seconds=cpu.ru_utime+cpu.ru_stime,
             peak_rss_kib=cpu.ru_maxrss, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None))
    put_json(out/"status.json", dict(status=status, completed_updates=update, target_updates=target,
             evaluation="pending_original_bank" if not args.smoke else "smoke_only"))
    print(out, status, update, flush=True)


def save_model(out, model, manifest, update):
    tmp = out/"checkpoint.tmp"
    torch.save(dict(manifest=manifest, update=update, state_dict=model.state_dict()), tmp)
    tmp.replace(out/"checkpoint.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-sboed-v6", action="store_true", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["implementation"] != "moe_sboed_v6" or cfg["system"] != "location_finding" or args.seed != cfg["seed"]:
        parser.error("V6 location config and matching seed required")
    name = args.config.stem + ("_smoke" if args.smoke else "")
    out = allocate_run(args.experiments, name, cfg["horizon"], [args.method])
    shutil.copyfile(args.config, out/"config_snapshot.yaml")
    print("OUTPUT", out, flush=True)
    put_json(out/"status.json", dict(status="initializing"))
    try:
        train(args, cfg, out)
    except BaseException as exc:
        put_json(out/"status.json", dict(status="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
