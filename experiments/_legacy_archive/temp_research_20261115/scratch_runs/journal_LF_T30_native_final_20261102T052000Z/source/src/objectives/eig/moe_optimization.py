"""Opt-in direct MoE optimizer; architecture and scientific objective are unchanged."""
import math
import torch


def make_optimizer(policy, args):
    router = list(policy.router.parameters())
    router_ids = {id(p) for p in router}
    experts = [p for p in policy.parameters() if id(p) not in router_ids]
    rates = [args.learning_rate, args.learning_rate * args.moe_router_lr_scale]
    optimizer = torch.optim.Adam([
        {'params': experts, 'lr': rates[0], 'name': 'experts'},
        {'params': router, 'lr': rates[1], 'name': 'router'},
    ])

    def set_learning_rates(update):
        # First actual update uses the initial rate; last uses the final fraction.
        progress = min(1., max(0., (update - 1) / max(args.updates - 1, 1)))
        factor = (args.moe_lr_final_fraction + (1 - args.moe_lr_final_fraction)
                  * .5 * (1 + math.cos(math.pi * progress))) if args.moe_lr_schedule == 'cosine' else 1.
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * factor

    return optimizer, set_learning_rates
