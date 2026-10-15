"""Adapter for pinned wgshen/vsOED Case 1a GMM-I training.

Unbounded runs additionally compress network inputs and check/clip gradients.
The model, posterior networks, replay sampling and actor/critic updates come
directly from the pinned upstream implementation.
"""
import json
import copy
import time
import torch
from torch import nn
from src.research.vendor.vsoed import PGvsOED, POST_APPROX
from src.research.vendor.vsoed.models import SOURCE
from src.research.vendor.vsoed.utils import set_random_seed

COMMIT = 'd0884f9ba30fbddc56618841c68af2f3bfe16539'


def compress_input(module,args):
    return (torch.asinh(args[0]),*args[1:])

def finite_gradient_step(optimizer,args,kwargs):
    parameters=[p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)

class ActionTrustRegionAdam(torch.optim.Adam):
    """Backtrack the offline Adam update in action space; no action bounds."""
    def __init__(self,parameters,actor,probe_inputs,radius=.05,**kwargs):
        super().__init__(parameters,**kwargs)
        self.actor,self.probe_inputs,self.radius=actor,probe_inputs,float(radius)
        if self.radius<=0:raise ValueError('Positive actor update radius required')
        self.last_scale=1.;self.last_change=0.;self.backtracks=0
    def step(self,closure=None):
        if closure is not None:raise ValueError('Closure unsupported')
        probes=self.probe_inputs()
        params=[p for g in self.param_groups for p in g['params']]
        old=[p.detach().clone() for p in params]
        old_state=copy.deepcopy(self.state_dict())
        with torch.no_grad():before=self.actor(probes).detach()
        if not torch.isfinite(before).all():raise FloatingPointError('Nonfinite reference actor designs')
        value=super().step()
        proposal=[p.detach().clone() for p in params]
        for attempt in range(25):
            scale=2.**(-attempt)
            with torch.no_grad():
                for p,a,b in zip(params,old,proposal):p.copy_(a+scale*(b-a))
                after=self.actor(probes)
                change=(after-before).norm(dim=-1).max()
            if torch.isfinite(change) and float(change)<=self.radius:
                self.last_scale,self.last_change=scale,float(change)
                self.backtracks+=attempt
                return value
        with torch.no_grad():
            for p,a in zip(params,old):p.copy_(a)
        self.load_state_dict(old_state)
        raise FloatingPointError('Actor line search cannot find a finite update')

class Table:
    def __init__(self): self.data = {}
    def record(self, key, value): self.data[key] = value
    def clear(self): self.data.clear()
    def __str__(self): return str(self.data)


class Logger:
    def __init__(self, every, directory=None):
        self.tabular = Table()
        self.logger = self
        self.every, self.history = every, []
        self.directory,self.algorithm=directory,None
    def push_prefix(self, prefix): pass
    def pop_prefix(self): pass
    def log(self, value): pass
    def dump_all(self):
        if 'Epoch' not in self.tabular.data: return
        row = dict(self.tabular.data)
        if self.algorithm is not None and isinstance(getattr(self.algorithm,'actor_optimizer',None),ActionTrustRegionAdam):
            opt=self.algorithm.actor_optimizer
            row.update(actor_trust_scale=opt.last_scale,actor_max_probe_change=opt.last_change,actor_backtracks=opt.backtracks)
        self.history.append(row)
        update = int(row['Epoch'])
        if update % self.every == 0:
            print(f"upstream vsOED-G-I update {update}: variational reward={row['Reward/MeanReward']}", flush=True)
            if self.directory is not None:
                (self.directory/'vsoed_upstream_history.json').write_text(json.dumps(self.history,indent=2)+'\n')
        if self.algorithm is not None and update>0 and update%1000==0:
            torch.save(dict(commit=COMMIT, update=update,
                actor_state={k:v.cpu() for k,v in self.algorithm.actor_net.state_dict().items()}),
                self.directory/f'vsoed_actor_update{update}.pt')


class UpstreamActor(nn.Module):
    kind = 'vsoed'
    outputs_design = True
    def __init__(self, actor, horizon, action_scale=4., obs_scale=3.):
        super().__init__()
        self.actor, self.horizon = actor, horizon
        self.action_scale, self.obs_scale = action_scale, obs_scale
    def forward(self, features, stage):
        # Upstream: one-hot stage, all designs, all observations; no masks.
        slots = features[:, 1:].reshape(len(features), self.horizon, 4)
        x = torch.zeros(len(features), self.horizon+3*(self.horizon-1), dtype=torch.float32)
        x[:, stage] = 1
        x[:, self.horizon:self.horizon+2*stage] = (slots[:, :stage, :2]*self.action_scale).flatten(1).float()
        begin = self.horizon+2*(self.horizon-1)
        x[:, begin:begin+stage] = (slots[:, :stage, 2]*self.obs_scale).float()
        return self.actor(x).double()


def train_upstream_vsoed(benchmark, horizon, settings, seed, directory, profile):
    if benchmark.name!='location_finding':raise ValueError('Upstream vsOED adapter supports location only')
    bounds=[(-float('inf'),float('inf'))]*2 if benchmark.design_bounds is None else [tuple(benchmark.design_bounds)]*2
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = dict(profile)
    trajectory_budget=cfg.pop('_trajectory_budget',None)
    if cfg.pop('_smoke', False):
        cfg.update(updates=2, new_trajectories=4, replay_batch=4, initial_replay=8,
                   replay_capacity=32, posterior_updates=1, critic_updates=1, log_every=1)
    generated=cfg['initial_replay']+cfg['updates']*cfg['new_trajectories']
    if trajectory_budget is not None and generated>trajectory_budget:
        raise ValueError('vsOED forward-trajectory plan exceeds common training cap')
    logger = Logger(cfg.get('log_every', 100),directory)
    started = time.perf_counter()
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device(device):
            set_random_seed(seed)
            model = SOURCE(multimodel=False, include_nuisance=False, include_goal=False)
            post = POST_APPROX(n_stage=horizon, n_design=2, n_obs=1, n_pois=[4],
                model_weight=0, poi_weight=1, goal_weight=0, n_mixture=8,
                mu_bounds={'poi': [[[-6, 6]]*4]}, max_sigmas={'poi': [[1.]*4]},
                truncnorm_info={'poi': [None]}, prior=model, n_incre=horizon,
                share_interm_net=False, use_NFs=False, activate=nn.ReLU,
                poi_post_lrs=[cfg['posterior_lr']]*horizon,
                poi_post_lr_scheduler_gammas=[cfg['lr_decay']]*horizon,
                log_every=cfg.get('log_every', 100), dowel=logger)
            algorithm = PGvsOED(horizon, 4, 2, 1, model, model, bounds,
                model.reward_fun, post.reward_fun, post_approx=post,
                actor_dimns=[256]*3, critic_dimns=[256]*3, activate=nn.ReLU)
            # Raw unbounded actor recursion needs a small initial output gain.
            # Preserve upstream hidden layers and all BA/actor-critic objectives.
            if benchmark.design_bounds is None:
                for actor_net in (algorithm.actor_net,algorithm.actor_target_net):
                    final=[m for m in actor_net.net.modules() if isinstance(m,nn.Linear)][-1]
                    with torch.no_grad():final.weight.mul_(.01);final.bias.zero_()
                algorithm.actor_target_net.load_state_dict(algorithm.actor_net.state_dict())
            if benchmark.design_bounds is None:
                for network in (algorithm.actor_net,algorithm.actor_target_net,algorithm.critic_net,algorithm.critic_target_net):
                    network.register_forward_pre_hook(compress_input)
                for nets in post.poi_post_nets.values():
                    for network in nets:network.feature_net.register_forward_pre_hook(compress_input)
                for optimizers in post.poi_post_optimizers.values():
                    for optimizer in optimizers:optimizer.register_step_pre_hook(finite_gradient_step)
            logger.algorithm=algorithm
            if benchmark.design_bounds is None:
                def actor_probes():
                    n=min(cfg.get('actor_trust_rows',32),len(algorithm.ds_hist))
                    return torch.cat([algorithm.form_actor_input(k,algorithm.ds_hist[:n,:k],algorithm.ys_hist[:n,:k]) for k in range(horizon)],0)
                actor_opt=ActionTrustRegionAdam(algorithm.actor_net.parameters(),algorithm.actor_net,actor_probes,radius=cfg.get('actor_trust_radius',.05),lr=cfg['actor_lr'])
            else:
                actor_opt=torch.optim.Adam(algorithm.actor_net.parameters(),lr=cfg['actor_lr'])
            critic_opt = torch.optim.Adam(algorithm.critic_net.parameters(), lr=cfg['critic_lr'])
            if benchmark.design_bounds is None:
                actor_opt.register_step_pre_hook(finite_gradient_step);critic_opt.register_step_pre_hook(finite_gradient_step)
            algorithm.train(n_update=cfg['updates'], n_newtraj=cfg['new_trajectories'],
                n_batch=cfg['replay_batch'], n_buffer_init=cfg['initial_replay'],
                n_buffer_max=cfg['replay_capacity'], buffer_device=torch.device(device),
                discount=cfg['discount'], n_critic_update=cfg['critic_updates'],
                n_post_approx_update=cfg['posterior_updates'],
                actor_optimizer=actor_opt, critic_optimizer=critic_opt,
                actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(actor_opt, cfg['lr_decay']),
                critic_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(critic_opt, cfg['lr_decay']),
                lr_target=cfg['target_rate'], design_noise_scale=cfg['exploration'],
                design_noise_decay=cfg['exploration_decay'], transition=cfg['transition'],
                on_policy=False, use_PCE=False, frozen=-1,
                log_every=cfg.get('log_every',100), dowel=logger, save_path=None)
            for net in [algorithm.actor_net, algorithm.critic_net]+[n for ns in post.poi_post_nets.values() for n in ns]:
                if not all(torch.isfinite(p).all() for p in net.parameters()):
                    raise FloatingPointError('Nonfinite upstream vsOED parameters')
            actor = algorithm.actor_net.cpu()
            # Net keeps design bounds as a plain tensor, rather than a buffer.
            actor.bounds = actor.bounds.cpu()
            actor.has_inf = actor.has_inf.cpu()
            checkpoint = dict(kind='vsoed', implementation='upstream_GMM_I', commit=COMMIT,
                horizon=horizon, actor_state=actor.state_dict(), profile=cfg,
                network_input_compression='asinh' if benchmark.design_bounds is None else 'identity',
                unbounded_actor_initial_gain=.01 if benchmark.design_bounds is None else None,
                critic_state={k:v.cpu() for k,v in algorithm.critic_net.state_dict().items()},
                posterior_states={str(k):[ {a:b.cpu() for a,b in n.state_dict().items()} for n in ns]
                                  for k,ns in post.poi_post_nets.items()})
            torch.save(checkpoint, directory/'vsoed.pt')
            (directory/'vsoed_upstream_history.json').write_text(json.dumps(logger.history,indent=2)+'\n')
    finally:
        torch.set_default_dtype(old_dtype)
    return UpstreamActor(actor,horizon,benchmark.action_scale,benchmark.obs_scale), dict(training='official_vsOED_GMM_I', upstream_commit=COMMIT,
        profile=cfg, device=device, selected_update=cfg['updates'], checkpoint_selection='final, as upstream',
        policy_parameters=sum(p.numel() for p in actor.parameters()), history=logger.history,
        training_seconds=time.perf_counter()-started,
        generated_training_trajectories=cfg['initial_replay']+cfg['updates']*cfg['new_trajectories'],
        contrast_training_trajectories=0,training_forward_trajectories=generated,training_trajectory_budget=trajectory_budget,
        reward_discount=cfg['discount'],implementation_adaptations=('Unbounded adaptation: invertible asinh network inputs, final actor initialization gain0.01, checked norm1 gradients; offline actor action-space line search; BA actor-critic unchanged; vendor unchanged' if benchmark.design_bounds is None else 'Common bounded action space; published raw network inputs and Adam actor updates; no added actor trust constraint; explicit episode cap; reward discount recorded separately; vendor unchanged'),
        posterior_optimizer_updates=cfg['updates']*cfg['posterior_updates'],
        critic_optimizer_updates=cfg['updates']*cfg['critic_updates'],
        actor_update_trust_radius=cfg.get('actor_trust_radius') if benchmark.design_bounds is None else None,
        actor_backtracks=getattr(actor_opt,'backtracks',0),
        replay_batch_size=cfg['replay_batch'],actor_optimizer_updates=cfg['updates'],
        forward_particle_stages=horizon*(cfg['initial_replay']+cfg['updates']*cfg['new_trajectories']))
