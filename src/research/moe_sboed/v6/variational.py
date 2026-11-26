"""Independent V6 actor trained with the pinned location vsOED-G-I algorithm."""
import copy
import json
import math
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.feature_policy import FeaturePolicy
from src.research.upstream_vsoed import COMMIT, Logger
from src.research.vendor.vsoed import PGvsOED, POST_APPROX
from src.research.vendor.vsoed.models import SOURCE
from src.research.vendor.vsoed.utils import set_random_seed


class NativeFeaturePolicy(FeaturePolicy):
    """Keep deployment in native actor precision while scoring likelihoods in FP64."""
    def start_history(self, sample):
        return super().start_history(sample.float())

    def update_history(self, state, token, stage):
        return super().update_history(state, token.float(), stage)


class PackedActor(nn.Module):
    """Map upstream mixed-stage replay rows to physical designs without future leakage."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def actions(self, rows):
        p = self.policy
        t = p.horizon
        stage = rows[:, :t].argmax(-1)
        index = torch.arange(t-1, device=rows.device)
        valid = index < stage[:, None]
        split = t + 2*(t-1)
        action = rows[:, t:split].reshape(-1, t-1, 2) / 4
        obs = (rows[:, split:].unsqueeze(-1) / 3).tanh()
        pair = torch.cat((action, obs), -1) * valid.unsqueeze(-1)
        previous = F.pad(pair[:, :-1], (0,0,1,0))
        tokens = p.token(torch.cat((pair, pair[:, :, :2] * pair[:, :, 2:], pair - previous), -1))
        age = (stage[:,None] - index).clamp_min(0)
        basis = p.basis[:-1].expand(len(rows),-1,-1)
        temporal = torch.cat((basis[:,:,:3], (-age.float()/5).exp().unsqueeze(-1)), -1)
        moments = (tokens.transpose(1,2) @ (temporal * valid.unsqueeze(-1))).transpose(1,2)
        history = p.history(F.pad(pair.flatten(1),(0,3)))
        ctx = p.context(torch.cat((F.silu(history),moments.flatten(1)/stage.clamp_min(1)[:,None]),-1))
        ctx = torch.cat((ctx,(t-stage[:,None]).to(ctx.dtype)/t),-1)
        z = torch.stack([expert(ctx) for expert in p.experts],1)
        if p.router is None:
            w = ctx.new_full((len(ctx),len(p.experts)),1/len(p.experts))
        else:
            w = p.router(torch.cat((ctx,z.flatten(1)),-1)).softmax(-1)
        logits = 2*p.head((z*w.unsqueeze(-1)).sum(1))
        return -4 + 8*logits.sigmoid()

    def forward(self, x):
        rows = x.reshape(-1,x.shape[-1])
        if len(rows) <= 2048:
            return self.actions(rows).reshape(*x.shape[:-1],2)
        # Native replay evaluates all stages in one actor update. Avoid retaining
        # 300k rows of expert activations simultaneously on a workstation GPU.
        chunks = [checkpoint(self.actions,r,use_reentrant=False) if torch.is_grad_enabled()
                  else self.actions(r) for r in rows.split(2048)]
        return torch.cat(chunks).reshape(*x.shape[:-1],2)


class BudgetReached(Exception):
    pass


def train_native(args, cfg, out, started):
    from src.research.moe_sboed.v6.policy import build_policy
    from src.research.moe_sboed.v6.train import put_json, save_model, sha
    import subprocess
    settings = cfg["training"]
    if (cfg["architecture"] != "independent_feature" or settings["initialization"] != "from_scratch"
            or cfg["location_finding"]["design_bounds"] != [-4,4] or cfg["observation"]["noise_sigma"] != .5):
        raise ValueError("Native Case 1a requires its bounded location model and an independent actor")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.smoke:
        raise RuntimeError("Production native training requires CUDA")
    profile = dict(settings["native"])
    updates = settings["updates"]
    if args.smoke:
        updates = 2
        profile.update(new_trajectories=4,replay_batch=4,initial_replay=8,replay_capacity=32,
                       posterior_updates=1,critic_updates=1)
    t = cfg["horizon"]
    dtype = torch.get_default_dtype()
    logger = Logger(100)
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device(device):
            set_random_seed(args.seed)
            source = SOURCE(multimodel=False, include_nuisance=False, include_goal=False)
            post = POST_APPROX(n_stage=t,n_design=2,n_obs=1,n_pois=[4],model_weight=0,poi_weight=1,
                goal_weight=0,n_mixture=8,mu_bounds={"poi":[[[-6,6]]*4]},max_sigmas={"poi":[[1.]*4]},
                truncnorm_info={"poi":[None]},prior=source,n_incre=t,share_interm_net=False,use_NFs=False,
                activate=nn.ReLU,poi_post_lrs=[profile["posterior_lr"]]*t,
                poi_post_lr_scheduler_gammas=[profile["lr_decay"]]*t,log_every=100,dowel=logger)
            a = PGvsOED(t,4,2,1,source,source,[(-4.,4.)]*2,source.reward_fun,post.reward_fun,
                        post_approx=post,actor_dimns=[256]*3,critic_dimns=[256]*3,activate=nn.ReLU)
            # Preserve the independently initialized native critic and posteriors.
            # Reset RNG after actor construction so Full and Uniform start with the same replay noise.
            set_random_seed(args.seed)
            model = build_policy(args.method,cfg,LocationFinding(cfg))
            a.actor_net = PackedActor(model)
            a.actor_target_net = copy.deepcopy(a.actor_net)
            set_random_seed(args.seed)
            actor = torch.optim.Adam(a.actor_net.parameters(),lr=profile["actor_lr"])
            critic = torch.optim.Adam(a.critic_net.parameters(),lr=profile["critic_lr"])
            manifest = dict(implementation="moe_sboed_v6",method=args.method,architecture=cfg["architecture"],
                horizon=t,seed=args.seed,initialization="from_scratch",objective=settings["objective"],
                upstream_commit=COMMIT,profile=profile,domain="bounded_location",target_updates=updates,
                experts=len(model.experts),config_sha256=sha(args.config),smoke=args.smoke,dtype="float32",
                parameters=sum(p.numel() for p in model.parameters()),torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name() if device.type=="cuda" else None,
                git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
            put_json(out/"manifest.json",manifest)
            print(json.dumps(manifest),flush=True)
            counts = dict(actor=0,critic=0,posterior=0)
            def counter(key):
                def counted(*args,**kwargs):
                    counts[key] += 1
                return counted
            for name,opt in [("actor",actor),("critic",critic),*[("posterior",o) for group in post.poi_post_optimizers.values() for o in group]]:
                opt.register_step_post_hook(counter(name))
            last_time = time.monotonic()
            last_update = 0
            def progress():
                nonlocal last_time,last_update
                update = int(a.update)
                if update <= last_update:
                    return
                if device.type=="cuda":
                    torch.cuda.synchronize()
                now=time.monotonic()
                reward=float(a.averaged_reward)
                if not math.isfinite(reward):
                    raise FloatingPointError("Nonfinite native variational reward")
                if update <= 2 or update % 25 == 0:
                    row=dict(update=update,variational_return=reward,elapsed_seconds=now-started,update_seconds=now-last_time)
                    with (out/"train_metrics.jsonl").open("a") as f:
                        print(json.dumps(row),file=f)
                    print(json.dumps(row),flush=True)
                if update==1 or update%100==0:
                    save_model(out,model,manifest,update)
                    put_json(out/"status.json",dict(status="running",completed_updates=update))
                step=now-last_time
                last_time,last_update=now,update
                if update<updates and now+max(60,1.5*step)>=started+settings["limit_seconds"]:
                    raise BudgetReached()
            logger.journal_callback=progress
            try:
                a.train(n_update=updates,n_newtraj=profile["new_trajectories"],n_batch=profile["replay_batch"],
                    n_buffer_init=profile["initial_replay"],n_buffer_max=profile["replay_capacity"],
                    buffer_device=device,discount=profile["discount"],n_critic_update=profile["critic_updates"],
                    n_post_approx_update=profile["posterior_updates"],actor_optimizer=actor,critic_optimizer=critic,
                    actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(actor,profile["lr_decay"]),
                    critic_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(critic,profile["lr_decay"]),
                    lr_target=profile["target_rate"],design_noise_scale=profile["exploration"],
                    design_noise_decay=profile["exploration_decay"],transition=profile["transition"],
                    on_policy=False,use_PCE=False,frozen=-1,log_every=100,dowel=logger,save_path=None)
            except BudgetReached:
                pass
            assert counts["actor"]==a.update and not a.use_PCE
            if not all(p.isfinite().all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite actor parameters")
            save_model(out,model,manifest,a.update)
            episodes=profile["initial_replay"]+profile["new_trajectories"]*a.update
            cpu=resource.getrusage(resource.RUSAGE_SELF)
            put_json(out/"budget.json",dict(updates=a.update,optimizer_steps=counts,inherited_episodes=0,
                fresh_training_episodes=episodes,simulated_observations=episodes*t,
                elapsed_seconds=time.monotonic()-started,cpu_seconds=cpu.ru_utime+cpu.ru_stime,
                peak_rss_kib=cpu.ru_maxrss,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else None))
            status="time_limited" if a.update<updates else "smoke_complete" if args.smoke else "complete"
            put_json(out/"status.json",dict(status=status,completed_updates=a.update,target_updates=updates))
    finally:
        torch.set_default_dtype(dtype)
