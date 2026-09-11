"""Posterior minimum safe control (MSC), distinct from MOCU sizing regret.

Probe policies minimize E_D[u_MSC(D;q)] at fixed posterior coverage q.
This does not certify out-of-sample safety or imply that information always
reduces expected quantiles. Report physical safety and oracle MSC separately.
"""
from .objective import posterior_msc, validate_msc_support
