"""Optional execution-only acceleration of the unchanged adaptive RK4 solver.

No state, time-step, precision, likelihood or gradient approximation is added.
Compilation overhead must be recorded separately from warm execution timings.
"""
import functools
import torch


def rk4_steps(y, interval, remaining, beta, gamma, population, *, steps):
    def rhs(value):
        incidence = beta * value[..., 0] * value[..., 1] / population
        recovery = gamma * value[..., 1]
        return torch.stack((-incidence, incidence - recovery, recovery), -1)
    for k in range(steps):
        step = (interval * (k < remaining)).unsqueeze(-1)
        a = rhs(y)
        b = rhs(y + step * a / 2)
        c = rhs(y + step * b / 2)
        d = rhs(y + step * c)
        y = y + step * (a + 2*b + 2*c + d) / 6
    return y


@functools.lru_cache(maxsize=32)
def compiled_steps(steps):
    # Distinct gradient modes and final partial blocks need several kernels.
    # This is a compiler-cache limit, not a numerical or training setting.
    torch._dynamo.config.recompile_limit=max(torch._dynamo.config.recompile_limit,64)
    return torch.compile(functools.partial(rk4_steps, steps=steps),
                         fullgraph=True, dynamic=True)


def compiled_block(y, interval, counts, beta, gamma, population, start, stop):
    # Eight steps keeps compiler graphs bounded. All original zero-length
    # padding steps are retained, including the final partial block.
    for offset in range(start, stop, 8):
        y = compiled_steps(min(8, stop-offset))(
            y.contiguous(), interval.contiguous(), (counts-offset).contiguous(),
            beta.contiguous(), gamma.contiguous(), population)
    return y
