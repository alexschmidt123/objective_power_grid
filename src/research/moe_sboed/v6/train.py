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
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a Slurm GPU allocation with CUDA")
    device = torch.device("cuda")
    settings = cfg["training"]
    horizon, seed = cfg["horizon"], cfg["seed"]
    if settings["initialization"] != "from_scratch" or settings["objective"] != "terminal_spce":
        raise ValueError("Expected fresh full-horizon sPCE training")
    batch, micro, contrasts = (settings[k] for k in ("batch_size", "microbatch_size", "contrasts"))
    if batch % micro:
        raise ValueError("Effective batch must be divisible by microbatch size")
    if args.smoke:
        batch, micro, contrasts = 8, 4, 16
    target = 2 if args.smoke else settings["updates"]
    bench = LocationFinding(cfg)
    torch.manual_seed(seed)
    model = build_policy(args.method, cfg, bench).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], betas=tuple(settings["adam_betas"]))
    rng = torch.Generator(device=device).manual_seed(seed)
    props = torch.cuda.get_device_properties(device)
    manifest = dict(method=args.method, implementation="moe_sboed_v6", seed=seed,
                    horizon=horizon, experts=len(model.experts), initialization="from_scratch",
                    architecture=cfg.get("architecture", "residual_sum"), parameters=sum(p.numel() for p in model.parameters()), train_contrasts=contrasts,
                    effective_batch=batch, microbatch=micro, target_updates=target,
                    dtype="float64", config_sha256=sha(args.config), smoke=args.smoke,
                    git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                    gpu=props.name, gpu_memory_bytes=props.total_memory, job_id=os.getenv("SLURM_JOB_ID"),
                    array_task=os.getenv("SLURM_ARRAY_TASK_ID"), node=os.uname().nodename)
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
            for _ in range(batch // micro):
                truth = torch.randn(micro, 4, generator=rng, dtype=torch.float64, device=device)
                negatives = torch.randn(micro, contrasts, 4, generator=rng, dtype=torch.float64, device=device)
                noise = torch.randn(micro, horizon, generator=rng, dtype=torch.float64, device=device)
                loss = -scores(bench, model, truth, negatives, noise, likelihood).sum() / batch
                loss.backward()
                loss_sum += loss.detach()
            terminal_loss = loss_sum.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
            if not math.isfinite(terminal_loss):
                raise FloatingPointError("Nonfinite terminal sPCE")
            optimizer.step()
            update += 1
            if update % settings["decay_every"] == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= settings["lr_decay"]
            torch.cuda.synchronize()
            update_seconds = time.monotonic() - tick
            if update <= 2 or update % 25 == 0:
                row = dict(update=update, spce=-terminal_loss, gradient_norm=norm.item(),
                           update_seconds=update_seconds, elapsed_seconds=time.monotonic()-START)
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
    put_json(out/"budget.json", dict(updates=update, fresh_training_episodes=update*batch,
             simulated_observations=update*batch*horizon, contrast_stage_likelihoods=update*batch*horizon*contrasts,
             inherited_episodes=0, elapsed_seconds=time.monotonic()-START, cpu_seconds=cpu.ru_utime+cpu.ru_stime,
             peak_rss_kib=cpu.ru_maxrss, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
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
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["implementation"] != "moe_sboed_v6" or cfg["system"] != "location_finding" or args.seed != cfg["seed"]:
        parser.error("V6 location config and matching seed required")
    name = args.config.stem + ("_smoke" if args.smoke else "")
    out = allocate_run("experiments", name, cfg["horizon"], [args.method])
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
