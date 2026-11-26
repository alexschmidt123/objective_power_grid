"""Replay development histories to measure policy-only deployment costs."""
import json
import time

import torch

from src.research.moe_sboed.v6.objective import trajectory


@torch.inference_mode()
def record_design_timing(out, model, bench, horizon):
    device = model.belief_particles.device
    rng = torch.Generator(device=device).manual_seed(98273)
    rows = {}
    for batch in (1, 16):
        truth = torch.randn(batch, 4, device=device, dtype=torch.float64, generator=rng)
        noise = torch.randn(batch, horizon, device=device, dtype=truth.dtype, generator=rng)
        actions, observations = trajectory(bench, model, truth, noise)
        tokens = torch.cat((bench.encode_action(actions), observations / bench.obs_scale), -1)
        def replay():
            state = model.start_history(tokens)
            for stage in range(horizon):
                bench.feasible(model.design_logits(state, stage), (), horizon)
                if stage + 1 < horizon:
                    state = model.update_history(state, tokens[:, stage], stage)
        replay()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(10):
            replay()
        torch.cuda.synchronize(device)
        seconds = (time.perf_counter() - start) / 10
        rows[batch] = dict(batch_episode_seconds=seconds,
                           mean_seconds_per_batched_design=seconds / horizon,
                           designs_per_second=batch * horizon / seconds)
    (out / "design_timing.json").write_text(json.dumps(dict(
        measurements=rows, repeats=10, warmup=1, horizon=horizon,
        gpu=torch.cuda.get_device_name(device),
        boundary="Synchronized wall time for policy initialization, sequential belief/history updates and feasible designs; replay observations in GPU memory; simulator and scorer excluded."
    ), indent=2) + "\n")
