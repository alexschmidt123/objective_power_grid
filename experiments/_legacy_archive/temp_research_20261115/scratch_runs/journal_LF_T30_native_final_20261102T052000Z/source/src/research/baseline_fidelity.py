"""Small, framework-independent decisions for original-method reproduction."""

def scorebed_reported_params(state, reference=False):
    """Official PolicyEIGBoundEvals reports EMA, not the raw optimizer iterate."""
    return state.params_ema if reference else state.params


def scorebed_restart_validation(plan, smoke=False):
    """Keep restart selection independent of development/final evaluation cases.

    Reuse the common high-resolution contrast count: L=16384 caps trajectory
    scores below 9.705 nats, below expected location-policy EIG in this study.
    This is a precision/resource choice, not a change to the EIG estimator.
    """
    config = dict(plan['validation'])
    if config['seed'] in {1001, 8831001}:
        raise ValueError('Restart selection requires independent validation draws')
    config.update(systems=4 if smoke else 128,
                  contrasts=8 if smoke else 1000000,
                  batch_size=4 if smoke else 16,
                  chunk=8 if smoke else 16384)
    return config


def rl_optimizer_epoch_completed(transitions, minimum_buffer):
    """Native SAC.train anneals entropy once per optimizer epoch, after warmup."""
    return transitions >= minimum_buffer
